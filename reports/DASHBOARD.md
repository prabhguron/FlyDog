# Brain dashboard and robot framework

Implemented after the initial training run. All previously trained artifact hashes are unchanged.

## Available now

- Local dashboard: http://127.0.0.1:8765
- Interactive anatomical soma view with the 128 modeled neurons and 6,500 passive context neurons.
- Actual checkpoint activations, computation stages, no-can comparisons, and action probabilities.
- Synthetic stimulus controls, uploaded-photo inference, and a browser-camera input option.
- Simulated dog adapter, physical-driver callback interface, speed bounds, latched stop, and independent 500 ms watchdog.
- Corrected-box and negative-image feedback capture, plus dataset export and detector fine-tuning commands.

## Verification

22 tests passed, including:

- Instrumentation preserves the checkpoint's original forward output and selected actions.
- Can-linked brightness equals the actual RMS state difference from a no-can observation; it is zero for the no-can input.
- Graph coordinates are aligned by exact neuron ID and the saved checkpoint's connectivity matches the displayed graph.
- Watchdog stops without further inference calls; stale input, invalid range, close obstacles, and driver errors stop movement.
- Latched stop requires clearing and re-arming.
- Synthetic dashboard inputs cannot drive a non-simulated movement adapter.
- Local API rejects foreign origins and malformed observations.
- Sample image detection feeds the same brain pipeline; stale-frame feedback is rejected.
- Feedback export preserves the validation/test split and excludes exact held-out image duplicates.

Browser checks confirmed: can on/off changes the displayed response; the mock driver receives forward commands; STOP latches; a real TACO photo is detected and stimulates the brain; drawing and saving a corrected box persists an annotation with test-photo training exclusion. JavaScript syntax check passed. Camera access was not activated on the user's device; the actual camera and physical dog remain untested.

## Meaning and limits

The dashboard displays the existing **128-neuron connectome-constrained network**, not a whole-brain spiking simulation. The gray background is anatomy, not simulated activity. Connections are schematic edges between real soma positions. No new neural training was performed for this dashboard. Feedback changes the reviewed dataset only; a separate supervised fine-tuning run is required to update recognition weights.

The actual Freenove movement code and calibration are still to be supplied. The default driver cannot move hardware. The framework provides a local Python driver boundary; a remote transport would also need robot-side authentication, sequence handling, and watchdog enforcement.

Start/restart from the repository:

```bash
source .venv/bin/activate
python scripts/prepare_brain_view.py
python scripts/brain_dashboard.py
```

See [robot integration](../docs/ROBOT_INTERFACE.md), [movement adapter](../examples/freenove_adapter.py), and [feedback training](../docs/FEEDBACK_TRAINING.md).
