"""Final untouched-seed comparison after exploratory runs; no model selection."""
import json
import os
from flycan.paths import ROOT,RUNS
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import torch
from stable_baselines3 import PPO
from train_navigation import evaluate

def main():
    torch.set_num_threads(2)
    outputs={}
    for label,name in [('fly','nav_fly_42_long'),('mlp','nav_mlp_42_long'),('shuffled','nav_shuffled_42')]:
        path=RUNS/name/'best_model.zip';model=PPO.load(path,device='cpu')
        result=evaluate(model,episodes=200,seed=30000)
        result['checkpoint']=str(path.relative_to(ROOT))
        result['parameters']=sum(p.numel() for p in model.policy.parameters())
        outputs[label]=result
        if label=='fly':
            with torch.no_grad():model.policy.features_extractor.w.zero_()
            outputs['fly_edges_zeroed']=evaluate(model,episodes=200,seed=30000)
    outputs['random']=evaluate(None,episodes=200,seed=30000)
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    (out/'navigation_results.json').write_text(json.dumps(outputs,indent=2))
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rollouts'} for k,v in outputs.items()},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=list(outputs)
    axes[0].bar(labels,[outputs[k]['success_rate']*100 for k in labels],color=['#247c89','#888888','#599b5c','#bb9a48','#bbbbbb'])
    axes[0].set(ylabel='Success (%)',ylim=(0,105),title='Approach and stop • 200 new seeds');axes[0].tick_params(axis='x',rotation=25)
    for label,name in [('fly','nav_fly_42_long'),('mlp','nav_mlp_42_long'),('shuffled','nav_shuffled_42')]:
        d=np.load(RUNS/name/'evaluations.npz');axes[1].plot(d['timesteps'],d['results'].mean(axis=1),label=label)
    axes[1].set(xlabel='Environment steps',ylabel='Validation reward',title='Training progress');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'navigation.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
