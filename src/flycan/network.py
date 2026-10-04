"""Fixed real connectivity, learned encoder/update/readout; no biological claim.
Graph message passing is recurrent WITHIN each observation (three rounds).
There is no hidden-state memory carried across environment steps.
"""
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from flycan.paths import DATA

class FlyFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,graph_path=None,mode='fly',channels=4):
        super().__init__(observation_space,features_dim=64)
        graph=np.load(graph_path or DATA/'connectome/graph.npz')
        w=graph['weights'].copy();self.nodes=len(w);self.channels=channels
        if mode=='shuffled':
            # Scramble postsynaptic endpoints separately within each presynaptic column.
            rng=np.random.default_rng(2026)
            for col in range(len(w)):rng.shuffle(w[:,col])
            w/=np.maximum(np.abs(w).sum(1,keepdims=True),1e-6)
        elif mode=='zero':w[:]=0
        elif mode!='fly':raise ValueError(mode)
        self.register_buffer('w',torch.tensor(w))
        self.encode=nn.Linear(observation_space.shape[0],self.nodes*channels)
        self.update=nn.Linear(channels,channels)
        self.readout=nn.Sequential(nn.Flatten(),nn.Linear(self.nodes*channels,64),nn.Tanh())
    def node_states(self,obs):
        """Encoder state followed by each actual graph-update state (B,N,C)."""
        h=torch.tanh(self.encode(obs)).reshape(-1,self.nodes,self.channels)
        states=[h]
        for _ in range(3):
            h=torch.tanh(self.update(torch.matmul(self.w,h)))
            states.append(h)
        return states
    def forward(self,obs):
        return self.readout(self.node_states(obs)[-1])
