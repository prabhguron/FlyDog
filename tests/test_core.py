import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from flycan.env import CanApproachEnv
from flycan.network import FlyFeatures

def test_environment():check_env(CanApproachEnv())

def test_success_requires_stop():
    e=CanApproachEnv(noise=False);e.reset(seed=1);e.pos=np.array([0.,0.]);e.target=np.array([.35,0.]);e.heading=0
    _,_,done,_,info=e.step(3)
    assert done and info['is_success']

def test_target_collision():
    e=CanApproachEnv(noise=False);e.reset(seed=1);e.pos=np.array([0.,0.]);e.target=np.array([.15,0.]);e.heading=0;e.speed=.08
    _,_,done,_,info=e.step(0)
    assert done and info['collision'] and not info['is_success']

def test_target_behind_is_hidden():
    e=CanApproachEnv(noise=False);e.reset(seed=1);e.pos=np.array([0.,0.]);e.target=np.array([-1.,0.]);e.heading=0
    assert np.array_equal(e.observe()[:3],np.zeros(3))

def test_graph_affects_output_and_trains_adapters():
    torch.manual_seed(1);e=CanApproachEnv();net=FlyFeatures(e.observation_space)
    x=torch.tensor([[1.,.3,.04,1.],[1.,-.3,.04,1.]])
    before=net(x);w=net.w.clone()
    before.square().sum().backward()
    assert net.encode.weight.grad.abs().sum()>0
    assert not net.w.requires_grad
    with torch.no_grad():net.w.zero_()
    after=net(x)
    assert not torch.allclose(before,after)
    assert torch.count_nonzero(w)>0

def test_camera_convention():
    from flycan.perception import observation_from_boxes
    left=observation_from_boxes([[.2,.5,.1,.2]],[.9],2)
    right=observation_from_boxes([[.8,.5,.1,.2]],[.9],2)
    assert left[1]>0 and right[1]<0
    assert observation_from_boxes([],[],2)[0]==0

def test_dataset_splits_and_labels():
    import json
    from flycan.paths import DATA
    m=json.loads((DATA/'taco_can/manifest.json').read_text());groups={};hashes={}
    for r in m['records']:
        if 'error' in r:continue
        assert groups.setdefault(r['batch'],r['split'])==r['split']
        assert hashes.setdefault(r['sha256'],r['split'])==r['split']
        for row in (DATA/'taco_can/labels'/r['split']/f"{r['id']:05}.txt").read_text().splitlines():
            cls,x,y,w,h=map(float,row.split())
            assert cls==0 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1

def test_graph_direction_uses_presynaptic_input(tmp_path):
    w=np.array([[0,0],[1,0]],dtype=np.float32)
    np.savez(tmp_path/'g.npz',weights=w,root_ids=np.array([1,2]))
    net=FlyFeatures(CanApproachEnv().observation_space,graph_path=tmp_path/'g.npz',channels=1)
    signal=torch.tensor([[[2.],[0.]]])
    assert torch.equal(torch.matmul(net.w,signal),torch.tensor([[[0.],[2.]]]))
