# First run

Train a detector, score it, look at what it found. Four commands, no mmcv.

## Install

```bash
pip install -e "py123detection[torch,images]"
export PY123D_DATA_ROOT=/data/py123d
```

`torch` brings PyTorch, Lightning and Hydra; `images` brings the image pipeline. Add `viz` for the
pictures in step 4.

## 1. Check the loop runs

```bash
py123det-train experiment=nuscenes_mini trainer=smoke images=petr_320
```

`trainer=smoke` is two batches of training and two of validation on the smallest input size. It
exists to fail in two minutes rather than two hours: if the data root, the rig or the taxonomy is
wrong, you find out here.

## 2. Train

```bash
py123det-train experiment=nuscenes taxonomy=nuscenes_10cls trainer=ddp trainer.devices=2
```

Checkpoints and configs go to `work_dirs/<run name>/`, and the run name is derived from the model,
the dataset mix and the taxonomy, so two runs never share a directory. `run.resume=last` continues
a run that was requeued.

What to change and why is in [Training without mmcv](training.md).

## 3. Score

```bash
py123det-eval experiment=nuscenes taxonomy=nuscenes_10cls \
  evaluation.checkpoint=work_dirs/petr_nuscenes_10cls/checkpoints/last.ckpt
```

One `metrics.json` per dataset, plus `predictions.jsonl` beside it. Scoring a mixed run gives one
score per dataset rather than one number for all of them, because a single number cannot say which
dataset moved.

The predictions file is the reason step 4 needs no GPU, and it means you can re-score with a
different class range or box cap without running the model again:

```bash
py123det-eval experiment=nuscenes taxonomy=nuscenes_10cls \
  evaluation.rescore_from=work_dirs/petr_nuscenes_10cls/eval
```

## 4. Look at it

```bash
py123det-viz bev work_dirs/petr_nuscenes_10cls/eval/nuscenes/predictions.jsonl -o frames/
```

Ground truth filled orange, predictions hollow blue, shaded by confidence. Add the camera images
beside it with `py123det-viz frame ... --experiment nuscenes`, or step through a scene in 3D with
`py123det-view`. See [Looking at what the model found](visualisation.md).

## What to read next

| | |
| --- | --- |
| [Training without mmcv](training.md) | the configuration groups, mixed-dataset runs, frame selection, memory |
| [Looking at what the model found](visualisation.md) | the three views and what each one answers |
| [Validation](../validation.md) | what is measured against the thesis, and how closely |
| [CLI reference](../cli.md) | every flag, generated from the parsers |
