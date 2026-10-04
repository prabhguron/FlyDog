"""Instrumented brain backend. Displayed signals are model states, not spikes."""
from typing import Protocol
import numpy as np
import torch
from stable_baselines3 import PPO
from flycan.env import ACTIONS
from flycan.network import FlyFeatures
from flycan.paths import DATA, RUNS

class BrainBackend(Protocol):
    """Future spiking simulators can implement this boundary using their own units."""
    def infer(self, observation: np.ndarray) -> dict: ...

class FlyPolicyBrain:
    def __init__(self, checkpoint=None):
        torch.set_num_threads(2)
        self.model=PPO.load(checkpoint or RUNS/'nav_fly_42_long/best_model.zip',device='cpu')
        self.model.policy.set_training_mode(False)
        self.net=self.model.policy.features_extractor
        if not isinstance(self.net,FlyFeatures):raise ValueError('Brain view requires a FlyFeatures checkpoint')
        graph=np.load(DATA/'connectome/graph.npz')
        if not np.allclose(self.net.w.cpu().numpy(),graph['weights']):
            raise ValueError('Checkpoint connectivity differs from displayed FlyWire graph')
        self.root_ids=[str(x) for x in graph['root_ids']]

    @torch.inference_mode()
    def infer(self, observation):
        obs=np.asarray(observation,dtype=np.float32)
        if obs.shape!=(4,) or not self.model.observation_space.contains(obs):
            raise ValueError('Observation must be finite [visible,bearing,width,range] within the model bounds')
        x=torch.as_tensor(obs[None])
        stages=self.net.node_states(x)
        no_can=x.clone();no_can[:,:3]=0
        baseline=self.net.node_states(no_can)
        distribution=self.model.policy.get_distribution(x).distribution
        probabilities=distribution.probs[0].cpu().numpy()
        deltas=[(s-b)[0].cpu().numpy() for s,b in zip(stages,baseline)]
        states=[s[0].cpu().numpy() for s in stages]
        return {'observation':obs.tolist(),'action':ACTIONS[int(np.argmax(probabilities))],
                'probabilities':dict(zip(ACTIONS,map(float,probabilities))),
                'activity':np.mean(np.abs(states[-1]),axis=1).tolist(),
                'can_response':np.sqrt(np.mean(deltas[-1]**2,axis=1)).tolist(),
                'signed_change':np.mean(deltas[-1],axis=1).tolist(),
                'stages':[np.sqrt(np.mean(d**2,axis=1)).tolist() for d in deltas],
                'stage_signs':[np.mean(d,axis=1).tolist() for d in deltas],
                'channels':states[-1].tolist(),
                'units':'dimensionless latent activation; RMS difference from no-can input',
                'whole_brain':False,'spiking':False}
