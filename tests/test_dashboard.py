import json
import time
import threading
import numpy as np
import pytest
import requests
import torch
from PIL import Image
from flycan.brain import FlyPolicyBrain
from flycan.robot import RobotBridge, MockDogDriver, CallbackDogDriver
from flycan.feedback import FeedbackStore
from flycan.dashboard import DashboardRuntime, DashboardServer

@pytest.fixture(scope='module')
def brain():return FlyPolicyBrain()

class Clock:
    def __init__(self):self.t=100.
    def __call__(self):return self.t

@pytest.fixture
def bridge():
    clock=Clock();b=RobotBridge(clock=clock,watchdog=False)
    yield b,clock
    b.close()

def test_trace_preserves_checkpoint_behavior(brain):
    x=torch.tensor([[1.,.25,.04,.5],[0.,0.,0.,.5]])
    net=brain.net
    with torch.no_grad():
        old=torch.tanh(net.encode(x)).reshape(-1,net.nodes,net.channels)
        for _ in range(3):old=torch.tanh(net.update(torch.matmul(net.w,old)))
        assert torch.equal(net(x),net.readout(old))
    for obs in x.numpy():
        frame=brain.infer(obs)
        action=int(brain.model.predict(obs,deterministic=True)[0])
        assert frame['action']==('forward','left','right','stop')[action]
        assert len(frame['activity'])==128 and len(frame['stages'])==4

def test_can_response_is_actual_difference(brain):
    absent=brain.infer([0,0,0,.5]);present=brain.infer([1,.2,.04,.5])
    assert max(absent['can_response'])==0
    a=np.array(present['channels']);b=np.array(absent['channels'])
    assert np.allclose(present['can_response'],np.sqrt(((a-b)**2).mean(axis=1)))
    assert np.mean(present['can_response'])>.01
    with pytest.raises(ValueError):brain.infer([1,float('nan'),0,.5])

def test_watchdog_disarms_and_stops(bridge):
    b,c=bridge;b.arm();b.submit('forward',1,c())
    assert b.status()['command']['linear_mps']==.08
    c.t+=.51;b.check_timeout()
    assert not b.armed and b.status()['driver']['linear_mps']==0 and b.reason=='Command timeout'

def test_emergency_stop_latches(bridge):
    b,c=bridge;b.arm();b.submit('left',1,c());b.emergency_stop()
    with pytest.raises(ValueError):b.arm()
    b.submit('forward',1,c());assert b.status()['command']['action']=='stop'
    b.clear_stop();assert not b.armed and not b.estop

def test_bad_inputs_and_obstacles_stop(bridge):
    b,c=bridge
    for action,front,age in [('forward',.1,0),('forward',float('nan'),0),('forward',1,1),('jump',1,0)]:
        b.arm();b.submit(action,front,c()-age)
        assert b.status()['command']['action']=='stop'
        assert b.status()['driver']['linear_mps']==0

def test_callback_adapter_units_and_stop():
    calls=[];driver=CallbackDogDriver(lambda v,w:calls.append((v,w)),lambda:calls.append('stop'))
    c=Clock();b=RobotBridge(driver,clock=c,watchdog=False)
    try:
        b.arm();b.submit('left',1,c());assert calls[-1]==(0.,.45)
        b.submit('right',1,c());assert calls[-1]==(0.,-.45)
        b.submit('forward',1,c());assert calls[-1]==(.08,0.)
        b.disarm();assert calls[-1]=='stop'
    finally:b.close()

def test_feedback_preserves_test_exclusion(tmp_path):
    store=FeedbackStore(tmp_path);im=Image.new('RGB',(80,50))
    result=store.save(im,[[.5,.5,.2,.2]],'dataset_sample',1,True)
    assert not result['training_eligible'] and not result['weights_updated']
    result2=store.save(im,[],'image',2,True)
    assert not result2['training_eligible']
    with pytest.raises(ValueError):store.save(im,[[.9,.9,.5,.5]],'image',3,True)
    with pytest.raises(ValueError):store.save(im,[],'image',3,False)

def test_new_negative_feedback(tmp_path):
    s=FeedbackStore(tmp_path);result=s.save(Image.new('RGB',(40,30),'green'),[],'camera',3,True)
    assert result['training_eligible']
    record=json.loads((tmp_path/'annotations'/f"{result['id']}.json").read_text())
    assert record['kind']=='negative' and record['boxes']==[]

def test_http_contract_and_origins(brain,tmp_path):
    rt=DashboardRuntime(brain=brain);rt.feedback_store=FeedbackStore(tmp_path)
    server=DashboardServer(('127.0.0.1',0),rt);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}';headers={'X-FlyCan':'1'}
    try:
        assert requests.get(url+'/api/graph').json()['nodes'][0]['root_id']==brain.root_ids[0]
        payload=dict(visible=True,bearing=0,width=.04,front_range_m=1.5)
        assert requests.post(url+'/api/stimulus',json=payload).status_code==403
        assert requests.post(url+'/api/stimulus',json=payload,headers=headers|{'Origin':'https://other.example'}).status_code==403
        r=requests.post(url+'/api/stimulus',json=payload,headers=headers);assert r.status_code==200 and r.json()['action']=='forward'
        assert requests.post(url+'/api/stimulus',json={'visible':True},headers=headers).status_code==400
        assert requests.post(url+'/api/feedback',json={'frame_sequence':0,'boxes':[],'confirmed_complete':True},headers=headers).status_code==400
    finally:server.shutdown();server.server_close();rt.close();thread.join()

def test_demo_cannot_drive_physical_adapter(brain):
    calls=[];d=CallbackDogDriver(lambda v,w:calls.append((v,w)),lambda:calls.append('stop'))
    bridge=RobotBridge(d,watchdog=False);rt=DashboardRuntime(brain,bridge)
    try:
        bridge.arm();rt.demo(dict(visible=True,bearing=0,width=.04,front_range_m=2))
        assert not bridge.armed and all(x=='stop' for x in calls)
    finally:rt.close()

def test_sample_detection_and_feedback_are_connected(brain,tmp_path):
    rt=DashboardRuntime(brain=brain);rt.feedback_store=FeedbackStore(tmp_path)
    try:
        result=rt.sample({'front_range_m':1.5})
        assert result['source']=='dataset_sample' and result['observation'][0]==1
        assert result['detections'] and len(result['can_response'])==128
        assert not result['robot']['armed']
        saved=rt.feedback({'frame_sequence':result['sequence'],'boxes':[[.5,.5,.2,.2]],'confirmed_complete':True})
        assert not saved['training_eligible']
        rt.demo(dict(visible=False,bearing=0,width=.04,front_range_m=1.5))
        with pytest.raises(ValueError):rt.feedback({'frame_sequence':result['sequence'],'boxes':[],'confirmed_complete':True})
    finally:rt.close()

def test_real_watchdog_thread_stops_without_inference():
    b=RobotBridge()
    try:
        b.arm();b.submit('forward',1,time.monotonic())
        # No further submit or state call drives the watchdog; it must run independently.
        time.sleep(.65)
        assert not b.armed and b.driver.linear==0 and b.reason=='Command timeout'
    finally:b.close()

def test_feedback_export_keeps_holdouts_out(tmp_path,monkeypatch):
    import importlib.util
    from flycan.paths import ROOT
    from flycan.feedback import pixel_hash
    spec=importlib.util.spec_from_file_location('prepare_feedback',ROOT/'scripts/prepare_feedback.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'DATA',tmp_path)
    base=tmp_path/'taco_can'
    for split,color in [('train','red'),('val','blue'),('test','yellow')]:
        p=base/'images'/split;p.mkdir(parents=True)
        Image.new('RGB',(32,32),color).save(p/'one.jpg')
    store=FeedbackStore(tmp_path/'feedback')
    store.save(Image.new('RGB',(30,30),'green'),[[.5,.5,.2,.2]],'camera',1,True)
    store.save(Image.new('RGB',(30,30),'black'),[],'camera',2,True)
    with Image.open(base/'images/test/one.jpg') as im:
        excluded=store.save(im,[],'image',3,True)['id']
    module.main()
    manifest=json.loads((tmp_path/'feedback_can/manifest.json').read_text())
    assert manifest['training_images']==3 and manifest['reviewed_feedback_images']==2
    assert excluded in manifest['excluded_ids']
    lines=(tmp_path/'feedback_can/train.txt').read_text().splitlines()
    assert all('/test/' not in x and '/val/' not in x for x in lines)
    for line in lines:
        if '/feedback/' in line:assert __import__('pathlib').Path(line.replace('/images/','/labels/')).with_suffix('.txt').exists()

def test_driver_exception_stops_and_disarms():
    stopped=[]
    def fail(v,w):raise RuntimeError('driver fault')
    b=RobotBridge(CallbackDogDriver(fail,lambda:stopped.append(True)),watchdog=False)
    try:
        b.arm()
        with pytest.raises(RuntimeError):b.submit('forward',1,time.monotonic())
        assert not b.armed and b.command.action=='stop' and stopped
    finally:b.close()
