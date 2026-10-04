"""Convert YOLO boxes to the navigation observation convention."""
import numpy as np
from flycan.env import FOV

def observation_from_boxes(xywhn,confidence,front_range_m,fov_radians=FOV,threshold=.4):
    if not np.isfinite(front_range_m) or front_range_m<0:raise ValueError('Invalid range')
    obs=np.array([0.,0.,0.,min(front_range_m/3,1)],dtype=np.float32)
    boxes=np.asarray(xywhn).reshape(-1,4);scores=np.asarray(confidence).reshape(-1)
    if len(boxes)!=len(scores):raise ValueError('Box/confidence length mismatch')
    valid=np.flatnonzero(scores>=threshold)
    if len(valid):
        # Deterministic single-target selection. A real deployment needs tracking.
        b=boxes[valid[np.argmax(scores[valid])]]
        angle=np.arctan((1-2*b[0])*np.tan(fov_radians/2))
        obs[:3]=[1,np.clip(angle/(FOV/2),-1,1),np.clip(b[2],0,1)]
    return obs
