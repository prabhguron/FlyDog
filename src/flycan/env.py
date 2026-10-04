"""Planar approach-and-stop task. Observations approximate detector output.
No camera rendering or robot physics; true target distance is reward-only.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

ACTIONS=('forward','left','right','stop')
FOV=np.pi/2

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

class CanApproachEnv(gym.Env):
    metadata={}
    def __init__(self, noise=True, max_steps=160):
        self.observation_space=spaces.Box(np.array([0,-1,0,0],np.float32),np.ones(4,np.float32))
        self.action_space=spaces.Discrete(4)
        self.noise=noise;self.max_steps=max_steps
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.pos=self.np_random.uniform(-1,1,2)
        self.target=self.np_random.uniform(-2.3,2.3,2)
        while np.linalg.norm(self.target-self.pos)<.8:
            self.target=self.np_random.uniform(-2.3,2.3,2)
        self.heading=self.np_random.uniform(-np.pi,np.pi)
        self.speed=self.np_random.uniform(.065,.105)
        self.turn=self.np_random.uniform(.14,.23)
        self.can_width=self.np_random.uniform(.055,.075)
        self.t=0
        return self.observe(),{}
    def distance(self):return float(np.linalg.norm(self.target-self.pos))
    def observe(self):
        delta=self.target-self.pos;d=self.distance()
        angle=wrap(np.arctan2(delta[1],delta[0])-self.heading)
        visible=abs(angle)<FOV/2 and d<5
        if self.noise and self.np_random.random()<.03:visible=False
        bearing=np.clip(angle/(FOV/2)+self.np_random.normal(0,.015 if self.noise else 0),-1,1) if visible else 0
        # Normalized image bbox width, using pinhole projection approximation.
        size=min(1,self.can_width/max(.02,2*d*np.cos(angle)*np.tan(FOV/2))) if visible else 0
        # Single forward range sensor against enclosing square walls.
        direction=np.array([np.cos(self.heading),np.sin(self.heading)])
        ranges=[((3 if u>0 else -3)-p)/u for p,u in zip(self.pos,direction) if abs(u)>1e-6]
        front=max(0,min(ranges))
        return np.array([float(visible),bearing,size,min(front/3,1)],np.float32)
    def step(self,action):
        before=self.distance(); self.t+=1
        if action==0:self.pos+=self.speed*np.array([np.cos(self.heading),np.sin(self.heading)])
        elif action==1:self.heading=wrap(self.heading+self.turn)
        elif action==2:self.heading=wrap(self.heading-self.turn)
        elif action!=3:raise ValueError(action)
        d=self.distance()
        collision=bool(np.any(np.abs(self.pos)>2.9) or d<.13)
        success=bool(action==3 and .20<=d<=.50)
        reward=4*(before-d)-.01
        if action==3 and not success:reward-=.05
        if success:reward+=8
        if collision:reward-=5
        terminated=success or collision
        truncated=self.t>=self.max_steps and not terminated
        return self.observe(),float(reward),terminated,truncated,{'is_success':success,'collision':collision,'distance':d}
