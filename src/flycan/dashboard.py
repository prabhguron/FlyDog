"""Local brain dashboard and frame-to-motion pipeline; mock driver by default."""
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
from PIL import Image, ImageOps
from flycan.brain import FlyPolicyBrain
from flycan.perception import observation_from_boxes
from flycan.robot import RobotBridge
from flycan.feedback import FeedbackStore
from flycan.paths import ROOT,DATA,RUNS

WEB=Path(__file__).parent/'web'
MAX_BODY=6*1024*1024

class DashboardRuntime:
    def __init__(self,brain=None,robot=None):
        self.brain=brain or FlyPolicyBrain();self.robot=robot or RobotBridge()
        self.lock=threading.RLock();self.detector=None;self.sequence=0
        self.feedback_store=FeedbackStore();self.last_image=None;self.last_image_sequence=None
        self.frame=self.brain.infer(np.array([0,0,0,.5],np.float32))
        self.frame.update(source='demo',detections=[],sequence=0,updated_at=time.time(),latency_ms=0)
        self.geometry=json.loads((DATA/'connectome/brain_view.json').read_text())
        if [n['root_id'] for n in self.geometry['nodes']]!=self.brain.root_ids:
            raise ValueError('Brain geometry neuron order differs from checkpoint')
    def snapshot(self):
        with self.lock:return self.frame|{'robot':self.robot.status(),'age_ms':round((time.time()-self.frame['updated_at'])*1000)}
    def observe(self,obs,front_range_m,source='demo',observed_at=None,detections=None,allow_motion=True):
        received=time.monotonic() if observed_at is None else observed_at
        with self.lock:
            frame=self.brain.infer(obs)
            if allow_motion:self.robot.submit(frame['action'],front_range_m,received)
            else:self.robot.disarm()
            self.sequence+=1
            self.frame=frame|{'source':source,'sequence':self.sequence,'detections':detections or [],
                              'updated_at':time.time(),'latency_ms':round((time.monotonic()-received)*1000,1)}
            return self.snapshot()
    def demo(self,payload):
        try:
            visible=payload['visible'];bearing=float(payload['bearing']);width=float(payload['width']);front=float(payload['front_range_m'])
        except (KeyError,TypeError,ValueError) as e:raise ValueError('Required: visible, bearing, width, front_range_m') from e
        if type(visible) is not bool or not all(map(math.isfinite,[bearing,width,front])) or not (-1<=bearing<=1 and 0<=width<=1 and 0<=front<=10):
            raise ValueError('Invalid stimulus values')
        obs=np.array([float(visible),bearing if visible else 0,width if visible else 0,min(front/3,1)],np.float32)
        return self.observe(obs,front,allow_motion=self.robot.driver.simulated)
    def image(self,payload,source_label=None):
        received=time.monotonic()
        try:
            front=float(payload['front_range_m']);live=payload.get('live',False)
            if not math.isfinite(front) or not 0<=front<=10 or type(live) is not bool:raise ValueError('Invalid range or live flag')
            raw=base64.b64decode(payload['image_base64'],validate=True)
            if len(raw)>4*1024*1024:raise ValueError('Image exceeds 4 MB')
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>16000000:raise ValueError('Image exceeds 16 megapixels')
                image=ImageOps.exif_transpose(im).convert('RGB')
        except (KeyError,TypeError,OSError) as e:raise ValueError('Invalid image request') from e
        with self.lock:
            if self.detector is None:
                for key,folder in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','matplotlib')]:
                    path=ROOT/'.cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ.setdefault(key,str(path))
                from ultralytics import YOLO
                self.detector=YOLO(RUNS/'detector/weights/best.pt')
            result=self.detector.predict(image,imgsz=320,device='cpu',verbose=False,conf=.4)[0]
            boxes=result.boxes.xywhn.cpu().numpy();confidence=result.boxes.conf.cpu().numpy()
            obs=observation_from_boxes(boxes,confidence,front)
            detections=[{'xywhn':b.tolist(),'confidence':float(c)} for b,c in zip(boxes,confidence)]
            result_frame=self.observe(obs,front,source_label or ('camera' if live else 'image'),received,detections,allow_motion=live and self.robot.driver.simulated)
            self.last_image=image.copy();self.last_image_sequence=self.sequence
            return result_frame
    def sample_bytes(self):
        manifest=json.loads((DATA/'taco_can/manifest.json').read_text())
        r=next(r for r in manifest['records'] if r['split']=='test' and r.get('boxes',0)>0)
        path=DATA/f"taco_can/images/test/{r['id']:05}.jpg"
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1280,1280))
            output=io.BytesIO();im.save(output,format='JPEG',quality=90)
        return output.getvalue()
    def sample(self,payload):
        return self.image({'image_base64':base64.b64encode(self.sample_bytes()).decode(),'front_range_m':payload['front_range_m']},'dataset_sample')
    def feedback(self,payload):
        with self.lock:
            if self.last_image is None or payload.get('frame_sequence')!=self.sequence or self.last_image_sequence!=self.sequence:
                raise ValueError('Frame changed; freeze and label the current image')
            return self.feedback_store.save(self.last_image,payload.get('boxes'),self.frame['source'],self.sequence,payload.get('confirmed_complete'))
    def close(self):self.robot.close()

class DashboardServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,runtime):self.runtime=runtime;super().__init__(address,Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send_data(self,status,data,kind='application/json'):
        if kind=='application/json':data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)))
        self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        try:self.wfile.write(data)
        except (BrokenPipeError,ConnectionResetError):pass
    def valid_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
    def do_GET(self):
        if not self.valid_host():return self.send_data(403,{'error':'Local host only'})
        rt=self.server.runtime
        if self.path=='/api/state':return self.send_data(200,rt.snapshot())
        if self.path=='/api/graph':return self.send_data(200,rt.geometry)
        if self.path=='/api/sample':return self.send_data(200,rt.sample_bytes(),'image/jpeg')
        assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
        if self.path in assets:
            name,kind=assets[self.path];return self.send_data(200,(WEB/name).read_bytes(),kind)
        self.send_data(404,{'error':'Not found'})
    def do_POST(self):
        if not self.valid_host() or self.headers.get('X-FlyCan')!='1':return self.send_data(403,{'error':'Local application request required'})
        origin=self.headers.get('Origin')
        if origin and origin not in {f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}:
            return self.send_data(403,{'error':'Origin rejected'})
        if self.headers.get('Content-Type')!='application/json':return self.send_data(415,{'error':'Use application/json'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=MAX_BODY:return self.send_data(413,{'error':'Request too large or empty'})
            p=json.loads(self.rfile.read(size))
            if not isinstance(p,dict):raise ValueError('JSON object required')
            rt=self.server.runtime
            if self.path=='/api/stimulus':return self.send_data(200,rt.demo(p))
            if self.path=='/api/image':return self.send_data(200,rt.image(p))
            if self.path=='/api/sample-detect':return self.send_data(200,rt.sample(p))
            if self.path=='/api/feedback':return self.send_data(200,rt.feedback(p))
            if self.path=='/api/robot':
                command=p.get('command')
                if command=='arm':rt.robot.arm()
                elif command=='disarm':rt.robot.disarm()
                elif command=='stop':rt.robot.emergency_stop()
                elif command=='clear':rt.robot.clear_stop()
                else:raise ValueError('Unknown robot command')
                return self.send_data(200,rt.robot.status())
            self.send_data(404,{'error':'Not found'})
        except (ValueError,TypeError) as e:self.send_data(400,{'error':str(e)})
        except Exception as e:
            self.server.runtime.robot.disarm()
            self.send_data(500,{'error':f'Pipeline failed: {type(e).__name__}: {e}'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8765);args=p.parse_args()
    runtime=DashboardRuntime();server=DashboardServer(('127.0.0.1',args.port),runtime)
    print(f'FlyCan brain view: http://127.0.0.1:{server.server_port} (simulated driver)',flush=True)
    try:server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:pass
    finally:server.server_close();runtime.close()

if __name__=='__main__':main()
