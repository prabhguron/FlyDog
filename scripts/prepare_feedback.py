"""Merge reviewed labels into a NEW training config, preserving original val/test."""
import json
from pathlib import Path
from PIL import Image, ImageOps
from flycan.feedback import pixel_hash
from flycan.paths import DATA

def main():
    base=DATA/'taco_can';feedback=DATA/'feedback';out=DATA/'feedback_can';out.mkdir(exist_ok=True)
    protected=set();train={}
    for split in ('train','val','test'):
        for path in sorted((base/'images'/split).glob('*.jpg')):
            with Image.open(path) as im:digest=pixel_hash(ImageOps.exif_transpose(im))
            if split=='train':train[digest]=path
            else:protected.add(digest)
    added=0;excluded=[]
    for path in sorted((feedback/'annotations').glob('*.json')):
        r=json.loads(path.read_text());digest=r['pixel_sha256']
        if not r.get('reviewed') or not r.get('training_eligible') or digest in protected:
            excluded.append(r['id']);continue
        image=feedback/'images'/f"{r['id']}.jpg"
        if not image.exists():raise FileNotFoundError(image)
        label=feedback/'labels'/f"{r['id']}.txt";label.parent.mkdir(exist_ok=True)
        label.write_text('\n'.join('0 '+' '.join(f'{v:.8f}' for v in b) for b in r['boxes']))
        train[digest]=image;added+=1
    if added==0:raise SystemExit('No eligible reviewed feedback yet. Test photos are excluded from training.')
    (out/'train.txt').write_text('\n'.join(str(p) for p in train.values())+'\n')
    (out/'dataset.yaml').write_text(f'path: {out}\ntrain: train.txt\nval: {base}/images/val\ntest: {base}/images/test\nnames:\n  0: can\n')
    (out/'manifest.json').write_text(json.dumps({'reviewed_feedback_images':added,'training_images':len(train),'excluded_ids':excluded,
        'validation_and_test':'original TACO split unchanged','limitation':'Exact pixel duplicates excluded; transformed near-duplicates still require human review'},indent=2))
    print(f'{added} reviewed feedback images; {len(train)} total training images. Dataset: {out}/dataset.yaml')

if __name__=='__main__':main()
