"""Human-reviewed detector labels; never applies online weight updates."""
import hashlib
import json
import time
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from flycan.paths import DATA

def pixel_hash(image):
    image=image.convert('RGB')
    return hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest()

class FeedbackStore:
    def __init__(self,path=None):self.path=Path(path or DATA/'feedback')
    def save(self,image,boxes,source,frame_sequence,confirmed_complete):
        if confirmed_complete is not True:raise ValueError('Confirm that every can is labeled')
        try:b=np.asarray(boxes,dtype=float).reshape(-1,4)
        except (ValueError,TypeError) as e:raise ValueError('Boxes must be [center_x, center_y, width, height]') from e
        if len(b)>100 or not np.isfinite(b).all():raise ValueError('Invalid boxes')
        if len(b) and ((b[:,2:]<=0).any() or (b[:,2:]>1).any() or (b[:,:2]-b[:,2:]/2<-.00001).any() or (b[:,:2]+b[:,2:]/2>1.00001).any()):
            raise ValueError('Boxes must lie within normalized image bounds')
        digest=pixel_hash(image);identity=digest
        for d in ('images','annotations','labels'):(self.path/d).mkdir(parents=True,exist_ok=True)
        target=self.path/'annotations'/f'{identity}.json'
        prior=json.loads(target.read_text()) if target.exists() else {}
        eligible=source!='dataset_sample' and prior.get('training_eligible',True)
        record={'id':identity,'pixel_sha256':digest,'source':source,'frame_sequence':frame_sequence,
                'boxes':b.tolist(),'reviewed':True,'training_eligible':eligible,'updated_at':time.time(),
                'width':image.width,'height':image.height,'kind':'positive' if len(b) else 'negative'}
        image.save(self.path/'images'/f'{identity}.jpg',quality=95)
        temp=target.with_suffix(f'.{uuid.uuid4().hex}.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(target)
        with (self.path/'events.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return {'id':identity,'training_eligible':eligible,'boxes':len(b),'weights_updated':False}
