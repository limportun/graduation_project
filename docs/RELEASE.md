# Release checklist

## Source release

1. Run `python -m pytest -q` in the reference environment.
2. Confirm `git status --short` contains only intended changes.
3. Confirm the repository tracks only the published pretrained checkpoint;
   training checkpoints, TensorBoard events, videos, caches, and editor files
   must remain ignored.
4. Create an annotated source tag, for example `v1.0.0`.
5. Create a GitHub Release from the tag.

## Repository-included pretrained checkpoint

The pretrained policy is committed directly to the repository at:

```text
checkpoint/STTRL_DVO/pretrained/ppo_multi_3000000_steps.zip
```

Before tagging, confirm that Git tracks exactly this checkpoint:

```bash
git ls-files checkpoint
```

The output must contain the pretrained path above and no `PPO_*` training
directories. Its expected SHA-256 is:

```text
2ef2006cdd7718096d64ccdbd9338f83075a7f0f1aa83ed4d3b468e4e99929e5
```

From a clean clone, recompute the hash, load the checkpoint on CPU, and run at
least one evaluation episode using the default checkpoint path:

```bash
python -m scripts.test --methods sttrl_dvo --seeds 1 --episodes 1 --device cpu
```

Do not upload a duplicate checkpoint as a GitHub Release asset. The GitHub
Release points to the tagged source tree, which already contains the model.

This project is a source-repository research release and is not uploaded to
PyPI.
