# Robot integration contract

The brain, visualization, and movement driver are separate modules. The current dashboard uses a **mock dog driver** and sends no hardware commands. The actual Freenove movement implementation is intentionally left to the code you will supply.

## Data flow

```
Camera frame + range sensor
   -> can detector (boxes + confidence)
   -> observation_from_boxes
   -> FlyPolicyBrain.infer
   -> RobotBridge.submit
   -> RobotDriver.apply
   -> your Freenove gait code
```

`FlyPolicyBrain` wraps the already trained PPO checkpoint. `BrainBackend` is the replacement boundary for a future spiking simulation. The current UI expects per-node activations, a no-can comparison, four computation stages, and an action distribution; a different simulator must provide its own compatible telemetry adapter and honest units. It is not currently running Shiu/Brian2 or a whole-brain model.

## Movement code to provide

Connect your existing forward, left, right, and stop calls through `examples/freenove_adapter.py:make_freenove_driver`. No retraining is needed to replace this driver. The `CallbackDogDriver` also accepts a single `move(linear_mps, angular_rps)` callback and a `stop()` callback.

| Field | Contract |
|---|---|
| `action` | `forward`, `left`, `right`, `stop` |
| `linear_mps` | Forward metres/second; current limit 0.08 |
| `angular_rps` | Radians/second; positive = left; current magnitude 0.45 |
| `sequence` | Increasing command number in this bridge instance |
| `ttl_ms` | 500 ms validity from receipt |
| `observed_at` | Local monotonic time when the sensor frame was captured/received; not wall-clock time |
| `front_range_m` | Current measured distance in metres; invalid/missing data must stop movement |

The vendor controller may use step lengths, servo angles, or arbitrary speed values instead of SI velocities. Calibrate that conversion in the driver. Do not feed these numbers directly into servo-angle APIs. Driver functions must return promptly, replacing the current gait setpoint instead of launching overlapping movement threads. The stop callback must interrupt ongoing movement.

`RobotBridge` starts disarmed, limits outputs to the above setpoints, stops below 0.20 m, rejects unknown actions and sensor timestamps older than 0.6 s, and runs a separate watchdog thread. More than 500 ms without a fresh accepted command stops and disarms. STOP is latched; clearing it leaves the robot disarmed until explicitly enabled again. `close()` stops the driver.

If movement code runs on another computer, implement the same TTL and stop behavior **on the robot**, plus sequence checking and authenticated transport. A laptop watchdog alone cannot stop a robot over a broken network. The current framework does not implement or claim to test a network-to-servo transport.

## Connect a sensor loop

```python
import time
from flycan.brain import FlyPolicyBrain
from flycan.robot import RobotBridge
from flycan.perception import observation_from_boxes
from examples.freenove_adapter import make_freenove_driver

# Supply calibrated, non-blocking functions from your movement code.
driver = make_freenove_driver(forward, turn_left, turn_right, stop)
brain = FlyPolicyBrain()
bridge = RobotBridge(driver)
try:
    bridge.arm()
    while running:
        captured_at = time.monotonic()
        # Supply fresh measured range and detections from your sensor code.
        boxes, confidences, range_m = read_sensors()
        observation = observation_from_boxes(boxes, confidences, range_m)
        result = brain.infer(observation)
        bridge.submit(result['action'], range_m, captured_at)
finally:
    bridge.close()
```

The undefined names here are your upcoming hardware functions, not existing package APIs. `boxes` are normalized `[center_x, center_y, width, height]`. Camera horizontal position is converted into positive-left model bearings. The current model assumes a 90° camera FOV and simplified can-width geometry. Calibrate camera parameters and test actual sensor observations before deployment. The current dashboard's range slider is simulated, so demo and browser-camera inputs cannot drive a non-simulated driver.

## Local dashboard API

Run `python scripts/brain_dashboard.py` and open http://127.0.0.1:8765. The server binds only to loopback. JSON POST requests require `Content-Type: application/json` and `X-FlyCan: 1`; foreign browser origins are rejected. This is a local development service, not a public hosting server.

| Endpoint | Purpose |
|---|---|
| `GET /api/graph` | Anatomical soma geometry, root IDs, graph edges |
| `GET /api/state` | Model activations, can-linked changes, probabilities, driver status |
| `POST /api/stimulus` | Synthetic observation: `visible`, `bearing`, `width`, `front_range_m` |
| `POST /api/image` | `image_base64`, `front_range_m`, optional `live` boolean; runs trained detector |
| `POST /api/sample-detect` | Detect a fixed held-out dataset example; accepts `front_range_m` |
| `POST /api/robot` | `command`: `arm`, `disarm`, `stop`, `clear` |
| `POST /api/feedback` | `frame_sequence`, corrected normalized `boxes`, `confirmed_complete: true` |

Snapshots do not drive movement. Continuous demo/camera updates can drive the mock driver; losing updates triggers its timeout. Root IDs are JSON strings to preserve 64-bit accuracy in JavaScript. Input images and feedback remain local.

## What the lights mean

The default view compares each modeled neuron's four-channel state with the same model given `[0, 0, 0, range]` (no detected can). Brightness is the RMS difference between those channel vectors. Teal/amber indicates the sign of the **mean channel change**, not excitatory/inhibitory physiology. The brightness display saturates at an RMS change of 0.3; the numerical readout retains actual values. Total activation instead shows mean absolute channel activation with a display saturation of 0.8. States are dimensionless; there are no spike timings or voltage measurements. The stage selector shows the encoder and three actual graph updates, not time propagation through a biological brain.

Positions come from the FlyWire annotation table's soma coordinates (4×4×40 nm voxels). The 128 modeled neurons are aligned by exact v783 root IDs. The additional 6,500 gray soma points are a deterministic sample of unmodeled anatomical context. Straight edges depict connectivity; they are not axons, dendrites, or reconstructed morphologies. The source URL, normalization, and SHA-256 are saved in `data/connectome/brain_view.json`.
