import argparse
import json
import time
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from flycan.env import CanApproachEnv
from flycan.network import FlyFeatures
from flycan.paths import DATA,RUNS

def evaluate(model,episodes=100,seed=20000):
    env=CanApproachEnv();results=[]
    rng=np.random.default_rng(seed)
    for i in range(episodes):
        obs,_=env.reset(seed=seed+i);ret=0
        for step in range(env.max_steps):
            a=int(rng.integers(4)) if model is None else int(model.predict(obs,deterministic=True)[0])
            obs,r,term,trunc,info=env.step(a);ret+=r
            if term or trunc:break
        results.append(dict(seed=seed+i,success=info['is_success'],collision=info['collision'],steps=step+1,reward=ret))
    return {'episodes':episodes,'success_rate':float(np.mean([r['success'] for r in results])),
            'collision_rate':float(np.mean([r['collision'] for r in results])),
            'mean_reward':float(np.mean([r['reward'] for r in results])), 'rollouts':results}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['fly','mlp','shuffled','zero'],default='fly')
    p.add_argument('--steps',type=int,default=100000);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume');p.add_argument('--name');p.add_argument('--eval-only');args=p.parse_args()
    torch.set_num_threads(2)
    out=RUNS/(args.name or f'nav_{args.mode}_{args.seed}')
    if (out/'config.json').exists() and not args.eval_only:
        raise FileExistsError(f'{out} already contains a run; choose a new --name')
    out.mkdir(parents=True,exist_ok=True)
    if args.eval_only:
        model=PPO.load(args.eval_only,device='cpu')
        (out/'evaluation.json').write_text(json.dumps(evaluate(model),indent=2));return
    env=make_vec_env(CanApproachEnv,n_envs=8,seed=args.seed)
    kwargs=dict(net_arch=dict(pi=[64],vf=[64]))
    if args.mode!='mlp':kwargs.update(features_extractor_class=FlyFeatures,features_extractor_kwargs=dict(mode=args.mode))
    if args.resume:model=PPO.load(args.resume,env=env,device='cpu')
    else:model=PPO('MlpPolicy',env,policy_kwargs=kwargs,learning_rate=3e-4,n_steps=128,batch_size=256,n_epochs=5,
                   gamma=.98,ent_coef=.01,seed=args.seed,device='cpu',verbose=1)
    config=vars(args)|{'actual_parameters':sum(p.numel() for p in model.policy.parameters()),
        'graph_sha256':hashlib.sha256((DATA/'connectome/graph.npz').read_bytes()).hexdigest(),
        'observation':['visible','bearing','bbox_width','front_range_normalized'],
        'limitations':['idealized detector observations','planar kinematics','no internal obstacles','single training seed']}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'untrained.json').write_text(json.dumps(evaluate(model,30,10000),indent=2))
    evalenv=make_vec_env(CanApproachEnv,n_envs=1,seed=10000)
    callbacks=[CheckpointCallback(save_freq=4096,save_path=str(out/'checkpoints')),
               EvalCallback(evalenv,best_model_save_path=str(out),log_path=str(out),eval_freq=4096,n_eval_episodes=20)]
    start=time.time();model.learn(total_timesteps=args.steps,callback=callbacks,reset_num_timesteps=not bool(args.resume))
    model.save(out/'final')
    best=PPO.load(out/'best_model.zip',device='cpu') if (out/'best_model.zip').exists() else model
    result=evaluate(best);result['training_seconds']=time.time()-start;result['timesteps']=model.num_timesteps
    (out/'evaluation.json').write_text(json.dumps(result,indent=2))
    (out/'random.json').write_text(json.dumps(evaluate(None),indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rollouts'},indent=2),flush=True)
    env.close();evalenv.close()

if __name__=='__main__':main()
