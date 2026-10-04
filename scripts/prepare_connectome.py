"""Create a bounded real-neuron FlyWire v783 subgraph, not a whole fly brain."""
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import requests
from flycan.paths import DATA

BASE='https://storage.googleapis.com/flywire-data/codex/data/fafb/783/'

def main():
    p=argparse.ArgumentParser(); p.add_argument('--nodes',type=int,default=128); args=p.parse_args()
    raw=DATA/'raw'; raw.mkdir(parents=True,exist_ok=True)
    sources={}
    for name in ('connections.csv.gz','classification.csv.gz'):
        file=raw/name
        if not file.exists():
            r=requests.get(BASE+name,timeout=120);r.raise_for_status();file.write_bytes(r.content)
        sources[name]={'url':BASE+name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    # Bound memory by streaming the 3.7M-edge table.
    parts=[]
    for c in pd.read_csv(raw/'connections.csv.gz',chunksize=200000):
        parts.append(c[c.neuropil.isin(['FB','EB','PB','NO'])])
    cx=pd.concat(parts,ignore_index=True)
    strength=pd.concat([cx.groupby('pre_root_id').syn_count.sum(),cx.groupby('post_root_id').syn_count.sum()],axis=1).fillna(0).sum(axis=1)
    ids=strength.sort_values(ascending=False,kind='stable').head(args.nodes).index.to_numpy(dtype=np.int64)
    ix={int(v):i for i,v in enumerate(ids)}
    # Keep ALL available neuropil edges between selected cells, not only CX contacts.
    W=np.zeros((len(ids),len(ids)),np.float32); count=0; nts={}
    for c in pd.read_csv(raw/'connections.csv.gz',chunksize=200000):
        c=c[c.pre_root_id.isin(ix)&c.post_root_id.isin(ix)]
        for r in c.itertuples(index=False):
            # Simplifying transmitter sign convention, not receptor-specific physiology.
            sign=-1 if r.nt_type in ('GABA','GLUT') else 1
            W[ix[int(r.post_root_id)],ix[int(r.pre_root_id)]]+=sign*r.syn_count
            count+=int(r.syn_count); nts[str(r.nt_type)]=nts.get(str(r.nt_type),0)+1
    W/=np.maximum(np.abs(W).sum(axis=1,keepdims=True),1)
    out=DATA/'connectome';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'graph.npz',weights=W,root_ids=ids)
    pd.read_csv(raw/'classification.csv.gz').query('root_id in @ids').to_csv(out/'neurons.csv',index=False)
    info={'release':783,'nodes':len(ids),'directed_edges':int(np.count_nonzero(W)),
          'synapses_before_normalization':count,'selection':'Top weighted-degree cells in FB/EB/PB/NO; induced graph across all neuropils',
          'orientation':'weights[post, pre]','normalization':'absolute incoming row sum',
          'transmitter_assumption':'GABA/GLUT negative; others positive; receptor effects not modeled',
          'transmitter_edge_rows':nts,'sources':sources,'whole_brain':False}
    (out/'manifest.json').write_text(json.dumps(info,indent=2));print(json.dumps(info,indent=2))

if __name__=='__main__':main()
