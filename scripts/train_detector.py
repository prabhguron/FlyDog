import argparse
import json
import os
from flycan.paths import DATA,RUNS,ROOT
(ROOT/'.cache/yolo').mkdir(parents=True,exist_ok=True)
(ROOT/'.cache/matplotlib').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/yolo'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import torch
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=20);p.add_argument('--device',default='cpu')
    p.add_argument('--imgsz',type=int,default=320);p.add_argument('--name',default='detector');p.add_argument('--resume')
    args=p.parse_args();torch.set_num_threads(2)
    os.chdir(ROOT)
    model=YOLO(args.resume or 'yolo11n.pt')
    model.train(data=str(DATA/'taco_can/dataset.yaml'),epochs=args.epochs,imgsz=args.imgsz,
                batch=8,device=args.device,workers=0,project=str(RUNS),name=args.name,
                seed=42,deterministic=True,patience=10,plots=True,save=True,exist_ok=False,
                resume=bool(args.resume),cache=False,amp=False)
    best=YOLO(model.trainer.best)
    metrics=best.val(data=str(DATA/'taco_can/dataset.yaml'),split='test',device=args.device,
                     imgsz=args.imgsz,workers=0,batch=8,project=str(RUNS),name=args.name+'_test')
    (model.trainer.save_dir/'test_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2))

if __name__=='__main__':main()
