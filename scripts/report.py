"""Record measured results and local checkpoints; no training or selection."""
import hashlib
import json
from pathlib import Path
from flycan.paths import ROOT,DATA,RUNS

def main():
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    nav=json.loads((out/'navigation_results.json').read_text())
    detector=json.loads((RUNS/'detector/test_metrics.json').read_text())
    graph=json.loads((DATA/'connectome/manifest.json').read_text())
    dataset=json.loads((DATA/'taco_can/manifest.json').read_text())
    artifacts={}
    for name,path in [('detector',RUNS/'detector/weights/best.pt'),('fly_policy',RUNS/'nav_fly_42_long/best_model.zip'),
                      ('mlp_policy',RUNS/'nav_mlp_42_long/best_model.zip'),('shuffled_policy',RUNS/'nav_shuffled_42/best_model.zip'),
                      ('graph',DATA/'connectome/graph.npz')]:
        artifacts[name]={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    (out/'artifacts.json').write_text(json.dumps(artifacts,indent=2))
    (out/'detector_metrics.json').write_text(json.dumps(detector,indent=2))
    (out/'graph_manifest.json').write_text(json.dumps(graph,indent=2))
    lines=['# Training results','',
           'Completed locally on Apple M2 CPU. This is a trained research prototype, not a hardware-ready dog controller.',
           '', '## Navigation','',
           'Each policy received approximately one million PPO environment steps. The fly and MLP runs used a 100k pilot followed by 900k additional steps; the shuffled control used one continuous run. All used training seed 42. Models were selected by validation reward, then evaluated on the same 200 previously unused environment seeds (30000–30199). Earlier 100-episode evaluations were exploratory and informed extending training.',
           '', '| Controller | Successful stops | Collisions | Mean reward | Parameters |',
           '|---|---:|---:|---:|---:|']
    for k,v in nav.items():
        lines.append(f"| {k} | {round(v['success_rate']*v['episodes'])}/{v['episodes']} | {round(v['collision_rate']*v['episodes'])}/{v['episodes']} | {v['mean_reward']:.2f} | {v.get('parameters','—')} |")
    lines+=['','The zeroed-edge test removes connectivity only at inference from the trained fly policy. It checks whether that policy uses its graph; it does not establish an advantage for biological topology. The shuffled control is not degree-preserving and the MLP is not parameter-matched. A single training seed cannot establish comparative superiority.',
            '', '![Navigation results](navigation.png)','', '## Can detector','',
            'YOLO11n was fine-tuned for 20 epochs at 320-pixel resolution. Best weights were selected on validation data; the numbers below are from the separate test split.', '']
    for k,v in detector.items():lines.append(f'- {k}: {v:.4f}')
    lines+=['','| Split | Images | Images with cans | Can boxes |','|---|---:|---:|---:|']
    for split in ('train','val','test'):
        rows=[r for r in dataset['records'] if r['split']==split and 'error' not in r]
        lines.append(f"| {split} | {len(rows)} | {sum(r['boxes']>0 for r in rows)} | {sum(r['boxes'] for r in rows)} |")
    lines+=['',f"{sum('error' in r for r in dataset['records'])} selected images were excluded because their downloaded aspect ratio disagreed with annotations. Failures are recorded in the data manifest.",
            '', 'Detector results are preliminary: the test set is small, and outdoor litter photos differ from a robot camera indoors. Navigation training used simulated detector-like features, not this detector in the loop. End-to-end physical success has not been measured.',
            '', '## FlyWire scope','',
            f"The policy uses {graph['nodes']} real v783 neurons and {graph['directed_edges']} directed aggregated connections representing {graph['synapses_before_normalization']:,} synapse counts before normalization. Connectivity is frozen; adapters and channel dynamics are trained. This is not the entire brain, a spiking simulation, or a biological preference-retraining result.",
            '', '## Verification and artifacts','',
            '- Eight tests passed: environment contract, stopping, collisions, hidden targets, trainable adapters, graph direction/effect, camera convention, and dataset splits/label ranges.',
            '- `artifacts.json` records checkpoint locations and SHA-256 hashes. Checkpoints and downloaded data remain locally available but are excluded from Git.',
            '- `navigation_results.json` includes every final evaluation episode; `detector_metrics.json` records detector test metrics.',
            '- See the README for setup, training, resuming, and offline image-to-action inference.',
            '', 'The repository is local. No GitHub repository has been published and no robot commands have been sent.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
