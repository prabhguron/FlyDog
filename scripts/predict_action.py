"""Offline image -> detector -> policy. Prints a proposal, never moves hardware."""
import argparse
import json
import os
from flycan.paths import ROOT
(ROOT/'.cache/yolo').mkdir(parents=True,exist_ok=True)
(ROOT/'.cache/matplotlib').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import torch
from ultralytics import YOLO
from stable_baselines3 import PPO
from flycan.perception import observation_from_boxes
from flycan.env import ACTIONS

def main():
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--detector',required=True)
    p.add_argument('--policy',required=True);p.add_argument('--front-range',type=float,required=True)
    p.add_argument('--confidence',type=float,default=.4);args=p.parse_args();torch.set_num_threads(2)
    result=YOLO(args.detector).predict(args.image,device='cpu',imgsz=320,verbose=False)[0]
    obs=observation_from_boxes(result.boxes.xywhn.cpu().numpy(),result.boxes.conf.cpu().numpy(),args.front_range,threshold=args.confidence)
    model=PPO.load(args.policy,device='cpu');action=int(model.predict(obs,deterministic=True)[0])
    proposal=ACTIONS[action]
    # Independent local guard for a future integration. No network/servo interface exists here.
    if args.front_range<.20:proposal='stop'
    print(json.dumps({'observation':obs.tolist(),'policy_action':ACTIONS[action],'guarded_proposal':proposal,'hardware_command_sent':False},indent=2))

if __name__=='__main__':main()
