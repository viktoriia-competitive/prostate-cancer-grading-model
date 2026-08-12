# Model Card — Prostate Cancer Grading Checkpoint

## Model

`best_model_finetuned.pt`

Six-class prostate cancer ISUP grade classification from pre-tiled histopathology images.

> **Research artifact only.** This checkpoint is not a medical device and is not intended for clinical diagnosis, treatment decisions, or clinical deployment without rigorous external validation and regulatory review.

## Inspected checkpoint metadata

The uploaded checkpoint was read using `torch.load(..., weights_only=True)`.

| Field | Value |
|---|---:|
| Saved epoch | 10 |
| Stored QWK | 0.6928080954 |
| State-dict tensors | 380 |
| Parameter elements | 26,919,177 |
| File size | 306.0 MiB |

## Input

```text
12 RGB tiles per sample
tile size: 128 × 128
tensor: [batch, 12, 3, 128, 128]
```

The checkpoint encodes a **156-D TDA input**:

```text
tda_branch.input_proj.0.weight -> [128, 156]
```

## Architecture encoded in this checkpoint

Observed state-dict shapes indicate a richer learned/gated fusion variant:

```text
12 tiles → ResNet50 → attention → 2048-D CNN representation
                                      +
                         156-D TDA → 128-D embedding
                                      ↓
                           2176-D fused vector
                                      ↓
                          classifier 512 → 256 → 6
```

The checkpoint also contains `tda_gate` and `tda_only_classifier` parameters.

## Compatibility note

The current `c.py` snapshot defines a simplified hybrid model with a 64-D TDA embedding and a smaller classifier. The supplied checkpoint therefore **does not represent the exact same class definition** as the current source snapshot.

For exact reproduction, use the matching model definition from the run that produced this checkpoint or adapt `c.py` to the checkpoint shapes above.

## Reported research result

The accompanying paper documents a separate 70/30 late-probability ensemble and reports a peak QWK of **0.7847 at epoch 12** on a **40-sample development validation subset**.

That value is distinct from the QWK stored in this uploaded checkpoint (0.6928).

## Intended use

Research, experimentation, education, and portfolio demonstration.

## Not intended for

Clinical diagnosis, treatment decisions, or autonomous medical workflow deployment.
