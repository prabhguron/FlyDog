# How corrective feedback improves recognition

The detector already receives loss penalties during supervised training: classification loss penalizes incorrect object predictions, and localization losses penalize inaccurate boxes. The navigation policy separately receives rewards for a correct approach/stop and penalties for collisions, time, and incorrect stops. Navigation reward does **not** backpropagate through the separately trained detector.

The live dashboard is inference-only. It never silently changes model weights. A change in brain brightness is an activation change, not learning.

## Record a missed can or false detection

1. Open Photo and upload an image, or stop a camera frame using **Correct boxes**.
2. For a missed can or inaccurate box, choose **Correct boxes**, then draw a rectangle around **every** can in the image. Predictions are cleared; the new boxes are a complete annotation, not additions to existing predictions.
3. Save corrected boxes. If the image has no cans at all, choose **No cans here** instead. If one of several detections is false, redraw only the real cans rather than labeling the entire image negative.

Corrections are stored in `data/feedback/images/` and `data/feedback/annotations/`, with an append-only event history. Repeated corrections for the same pixel image update its current annotation. Labels are human-reviewed examples; the system cannot know whether your labels are correct.

The built-in TACO sample is from the test split. Its corrections are saved for audit but explicitly excluded from training. The export script also excludes exact pixel duplicates of the original validation and test images. Resized or altered near-duplicates still require human review. Do not upload variations of validation/test images as new training examples.

## Retrain using reviewed feedback

Once you have several varied, correctly labeled camera examples:

```bash
python scripts/prepare_feedback.py
python scripts/train_detector.py \
  --data data/feedback_can/dataset.yaml \
  --weights runs/detector/weights/best.pt \
  --epochs 20 --name detector_feedback_v1
```

The export combines reviewed feedback with the original training images and preserves the existing validation/test split. Existing images are retained to reduce forgetting; images with no cans become negative training examples. There is no arbitrary extra punishment multiplier and no automatic weight update after a button click.

Compare the new run against the saved baseline on validation data, and reserve a separate robot-camera session for final evaluation. Track precision (false detections), recall (missed cans), and localization quality. Do not repeatedly tune on the final test results. The dashboard currently loads `runs/detector/weights/best.pt`; deployment of a new checkpoint is a deliberate configuration change, not automatic promotion.
