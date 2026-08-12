# Model checkpoint

`best_model_finetuned.pt` is approximately **306.0 MiB**, so it is not duplicated inside the lightweight ZIP.

Use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.pth"
git add .gitattributes
git add best_model_finetuned.pt
git commit -m "Add trained prostate grading checkpoint"
git push
```

See [`../MODEL_CARD.md`](../MODEL_CARD.md) for metadata and compatibility notes.
