"""Build anatomical soma geometry for the 128-node controller and passive context."""
import hashlib
import json
import numpy as np
import pandas as pd
import requests
from flycan.paths import DATA

URL='https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv'

def main():
    path=DATA/'raw/neuron_annotations.tsv';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        r=requests.get(URL,timeout=120);r.raise_for_status();path.write_bytes(r.content)
    ann=pd.read_csv(path,sep='\t',dtype={'root_id':str},low_memory=False).set_index('root_id')
    graph=np.load(DATA/'connectome/graph.npz');ids=[str(x) for x in graph['root_ids']]
    # Public annotation columns are FAFB voxels at 4 x 4 x 40 nm.
    columns=['soma_x','soma_y','soma_z'];valid=ann.dropna(subset=columns)
    xyz=valid[columns].to_numpy(float)*np.array([.004,.004,.040]) # micrometres
    center=np.median(xyz,axis=0);scale=np.max(np.percentile(xyz,99.5,axis=0)-np.percentile(xyz,.5,axis=0))/2
    positions=(xyz-center)/scale
    lookup={rid:pos for rid,pos in zip(valid.index,positions)}
    nodes=[]
    for i,rid in enumerate(ids):
        if rid not in lookup:raise ValueError(f'Missing soma for selected neuron {rid}; no invented position')
        r=ann.loc[rid]
        nodes.append({'index':i,'root_id':rid,'position':np.round(lookup[rid],5).tolist(),
                      'cell_type':str(r.get('cell_type','')) if pd.notna(r.get('cell_type')) else 'Unclassified',
                      'side':str(r.get('side','')),'transmitter':str(r.get('top_nt',''))})
    passive=[i for i,rid in enumerate(valid.index) if rid not in set(ids)]
    rng=np.random.default_rng(42);chosen=rng.choice(passive,min(6500,len(passive)),replace=False)
    W=graph['weights'];post,pre=np.nonzero(W)
    result={'nodes':nodes,'context':np.round(positions[chosen],4).tolist(),
            'edges':[{'source':int(a),'target':int(b),'weight':round(float(W[b,a]),5)} for a,b in zip(pre,post)],
            'metadata':{'dataset':'FlyWire FAFB v783','source':URL,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'coordinate_units':'FAFB voxel 4,4,40 nm converted to micrometres; centered and uniformly scaled',
                        'center_um':center.tolist(),'scale_um':float(scale),
                        'context_count':len(chosen),'context_simulated':False,
                        'meaning':'Soma positions. Straight edges show graph connections, not neuron morphology.'}}
    out=DATA/'connectome/brain_view.json';out.write_text(json.dumps(result,separators=(',',':')))
    print(f'Prepared {len(nodes)} modeled somas, {len(chosen)} passive somas, {len(pre)} connections -> {out}')

if __name__=='__main__':main()
