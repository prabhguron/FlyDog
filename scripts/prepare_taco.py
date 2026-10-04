"""Download official TACO annotations and a can/negative image subset.
Splits are by original batch, before download; no silent split reassignment.
"""
import concurrent.futures as cf
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import requests
from PIL import Image
from flycan.paths import DATA

URL = 'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json'

def main():
    raw = DATA / 'raw'; raw.mkdir(parents=True, exist_ok=True)
    ann = raw / 'taco_annotations.json'
    if not ann.exists():
        r = requests.get(URL, timeout=60); r.raise_for_status(); ann.write_bytes(r.content)
    d = json.loads(ann.read_text())
    cat_ids = {c['id'] for c in d['categories'] if c['name'] in ('Drink can', 'Food Can')}
    boxes = defaultdict(list)
    for a in d['annotations']:
        if a['category_id'] in cat_ids: boxes[a['image_id']].append(a['bbox'])
    positives = [im for im in d['images'] if im['id'] in boxes]
    negatives = [im for im in d['images'] if im['id'] not in boxes]
    rng = random.Random(42); rng.shuffle(negatives)
    selected = positives + negatives[:len(positives)]
    groups = sorted({im['file_name'].split('/')[0] for im in selected})
    rng.shuffle(groups)
    n = len(groups); a = max(1, int(n*.7)); b = max(a+1, int(n*.85))
    split = {g: ('train' if i<a else 'val' if i<b else 'test') for i,g in enumerate(groups)}
    out = DATA / 'taco_can'
    for s in ('train', 'val', 'test'):
        (out/'images'/s).mkdir(parents=True, exist_ok=True)
        (out/'labels'/s).mkdir(parents=True, exist_ok=True)
    def download(im):
        s = split[im['file_name'].split('/')[0]]
        dest = out/'images'/s/f"{im['id']:05}.jpg"
        url = im.get('flickr_640_url') or im['flickr_url']
        try:
            if not dest.exists():
                r=requests.get(url, timeout=25); r.raise_for_status()
                image=Image.open(io.BytesIO(r.content)).convert('RGB')
                # Normalized coordinates remain valid for resized source images.
                if abs(image.width/image.height - im['width']/im['height']) > .03:
                    raise ValueError('aspect ratio differs from annotation')
                image.save(dest, quality=92)
            with Image.open(dest) as check: check.verify()
            rows=[]
            for x,y,w,h in boxes[im['id']]:
                rows.append(f"0 {(x+w/2)/im['width']:.8f} {(y+h/2)/im['height']:.8f} {w/im['width']:.8f} {h/im['height']:.8f}")
            (out/'labels'/s/f"{im['id']:05}.txt").write_text('\n'.join(rows))
            return dict(id=im['id'], batch=im['file_name'].split('/')[0], split=s,
                        source_url=url, original=im, boxes=len(rows), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
        except Exception as e:
            return dict(id=im['id'], split=s, source_url=url, error=str(e))
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        records=list(pool.map(download,selected))
    manifest={'annotation_url':URL,'annotation_sha256':hashlib.sha256(ann.read_bytes()).hexdigest(),
              'seed':42,'classes':['Drink can','Food Can'],'split_method':'original batch grouped 70/15/15',
              'records':records}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'dataset.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: can\n')
    counts=Counter(r['split'] for r in records if 'error' not in r)
    print(json.dumps({'images':dict(counts),'failed':sum('error' in r for r in records),
                      'positive_images':len(positives),'selected':len(selected)},indent=2),flush=True)
    for s in ('train','val','test'):
        assert any(r['split']==s and r.get('boxes',0)>0 for r in records),f'No positives in {s}'
    # Exact image duplication must never span splits.
    hashes={}
    for r in records:
        if 'sha256' in r:
            assert hashes.setdefault(r['sha256'],r['split'])==r['split'],'Cross-split duplicate'

if __name__=='__main__': main()
