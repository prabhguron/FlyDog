# FlyCan: train a connectome-constrained can-seeking controller

A local research repository for a future Freenove robot-dog interface. It trains two separate components:

1. **Can detector:** YOLO11n fine-tuned on TACO Drink can + Food Can bounding boxes, merged into one class.
2. **Navigation:** PPO on a planar approach-and-stop environment, using a real FlyWire FAFB v783 subgraph as fixed connectivity.

This is **not a complete fly-brain emulation, a learned biological fruit preference, or a reproduction of FlyGM**. The initial controller has 128 actual FlyWire neurons. Its input encoding, channel dynamics, and action decoding are engineered and trained. It does not control dog joints or send hardware commands.

## Setup

Python 3.12 was used on an Apple M2 with 16 GiB RAM. The current session exposes CPU only.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
python scripts/prepare_taco.py
python scripts/prepare_connectome.py --nodes 128
pytest -q
```

Run commands from this repository directory. The lock file records the exact installed packages. It is an Apple Silicon snapshot; another platform may require resolving compatible wheels from `pyproject.toml`.

## Training

```bash
python scripts/train_detector.py --epochs 20 --imgsz 320 --device cpu
python scripts/train_navigation.py --mode fly --steps 1000000
python scripts/train_navigation.py --mode mlp --steps 1000000
python scripts/train_navigation.py --mode shuffled --steps 1000000
```

Detector runs automatically choose a new output directory when an existing name is occupied. Navigation output names must be unique when preserving experiments; use `--name my_experiment`. Navigation `--resume PATH` loads a PPO checkpoint and trains for the specified number of **additional** steps:

```bash
python scripts/train_navigation.py --mode fly --steps 900000 \
  --resume runs/nav_fly_42/final.zip --name nav_fly_42_long
```

Detector interruption recovery: `python scripts/train_detector.py --resume runs/detector/weights/last.pt`. A completed run's stripped checkpoint may not be resumable; load it as pretrained weights in a new training run instead.

Each navigation run saves configuration, checkpoints, a validation-selected `best_model.zip`, `final.zip`, random-policy evaluation, and 100 held-out seeded rollout results. Validation uses seeds beginning at 10000 and test uses seeds 20000–20099. Selection is based on validation reward, not test results. PPO rollouts are rounded to a multiple of 1024 environment steps. A single-seed result is not evidence that biological connectivity outperforms other architectures.

## What is trained

```text
camera image -> YOLO can box -> [visible, bearing, bbox width, front range]
    -> learned encoder -> three rounds of fixed-graph message passing
    -> learned readout -> PPO actor -> forward / left / right / stop
```

`src/flycan/network.py` uses `weights[post, pre]`. Synapse counts are aggregated and incoming absolute weights normalized. GABA and glutamate are assigned negative signs; other transmitters are positive. These are deliberately simplified assumptions, not receptor-aware physiology. The adjacency is a frozen buffer. Learned parameters include the encoder, shared channel update, readout, actor, and critic. There is no direct observation-to-action bypass around the graph.

Graph selection ranks neurons by weighted degree in FB, EB, PB and NO, then retains all released connections between the selected IDs. `data/connectome/neurons.csv` preserves IDs/classification and the manifest records source URLs and SHA-256 hashes. Selection does not establish a functionally complete navigation circuit. The matrix is dense for this small experiment: **do not set `--nodes` to whole-brain scale**; that requires a sparse implementation and a different compute plan.

The shuffled control permutes each presynaptic column's endpoints and renormalizes; it is not a degree-preserving rewiring. The MLP has a different parameter count, recorded in the configuration. For a scientific comparison, use matched parameter budgets and multiple seeds.

## Dataset and evaluation

Official source: https://github.com/pedropro/TACO

The preparation script selects every image annotated with Food Can or Drink can and an equally sized random set of images without these labels. It uses the source's 640-pixel Flickr variant; normalized COCO boxes are converted to YOLO format. Invalid aspect ratios are rejected and recorded. Source batches are split approximately 70/15/15 before downloading, and cross-split exact duplicate hashes are checked. Batch grouping reduces leakage but cannot guarantee different scenes or photographers. Test images are never used for detector training or checkpoint selection.

The image manifest retains each image's source metadata and failure reason. Raw photos, external model weights, large data, and runs are gitignored. The repository's MIT code license does not relicense TACO images, FlyWire data, or Ultralytics. TACO's downloaded annotation file has empty license metadata; consult original image terms before redistributing images. Ultralytics has its own AGPL/enterprise licensing terms.

## Scope of the navigation result

The PPO environment generates **idealized detector-like observations**, not images. TACO trains the detector separately. A high navigation score does not demonstrate an end-to-end camera-controlled robot.

The arena is a 6-by-6 metre square with boundary collisions, one target, a 90-degree field of view, randomized speed/turn rate/can width, and occasional missed detections. A correct stop is 0.20–0.50 m from the target; driving within 0.13 m is a collision. Ground-truth distance is used in the reward, not given as an observation. There are no interior obstacles, uneven terrain, gait dynamics, latency model, or persistent policy memory.

Before robot deployment: record robot-camera examples, calibrate camera FOV and bounding-box size, measure forward/turn commands, account for ultrasonic returns from the can itself, add tracking, test obstacles, and validate the complete closed loop. Keep Freenove's movement controller below the policy; include a local command timeout and stop control.

## Offline inference

```bash
python scripts/predict_action.py path/to/image.jpg \
  --detector runs/detector/weights/best.pt \
  --policy runs/nav_fly_42_long/best_model.zip \
  --front-range 1.5
```

This prints an action proposal only. Left/right camera coordinates are converted to the simulator's angle convention. It is an integration test, not a robot driver or a reliable decision about physical distance from a single arbitrary photo.

## Sources

- FlyWire explorer: https://codex.flywire.ai/?dataset=fafb
- FlyWire v783 data: https://storage.googleapis.com/flywire-data/codex/data/fafb/783/connections.csv.gz
- Reference spiking model (not the implementation used here): https://github.com/philshiu/Drosophila_brain_model
- FlyGM research inspiration: https://sites.google.com/view/flygm
- PPO implementation: https://github.com/DLR-RM/stable-baselines3
- Detector framework: https://github.com/ultralytics/ultralytics
- Freenove hardware software: https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi
