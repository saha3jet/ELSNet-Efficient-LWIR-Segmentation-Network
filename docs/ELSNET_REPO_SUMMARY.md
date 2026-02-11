# ELSNet Repo Summary

This document summarizes the ELSNet integration work and what remains before publishing.

## Implemented

- Backbone: `mmseg/models/backbones/elsnet.py`
  - SDM inserted before stem.
  - BEM inserted on D-branch stages.
  - Optional MSFM path integrated.
  - NNS generator included.
  - `forward_from_denoised` added for training orchestration.
- Decode head: `mmseg/models/decode_heads/els_head.py`
  - PID-style behavior preserved with ELS naming.
- Segmentor: `mmseg/models/segmentors/els_encoder_decoder.py`
  - Adds `loss_imse` orchestration while preserving EncoderDecoder flow.
- Losses: `mmseg/models/losses/lwir_losses.py`
  - `IMSELoss`, `LowSemanticLoss`, `BoundarySemanticLoss`.
  - `IMSELoss` supports inverse wavelet-domain mode.
- Registries updated:
  - `mmseg/models/backbones/__init__.py`
  - `mmseg/models/decode_heads/__init__.py`
  - `mmseg/models/segmentors/__init__.py`
  - `mmseg/models/losses/__init__.py`

## Configs

- Main config:
  - `configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py`
- LWIR dataset base config:
  - `configs/_base_/datasets/lwir_cityscapes_1024x1024.py`

## Environment Gap (Current Machine)

- Training smoke test could not run due to missing runtime packages:
  - `torch`
  - `mmengine`
  - `mmcv`

## Pre-Publish Checklist

1. Install runtime dependencies and run 1-iter smoke train.
2. Verify loss keys appear and are finite:
   - `loss_imse`
   - `decode.loss_sem_p`
   - `decode.loss_sem_i`
   - `decode.loss_bd`
   - `decode.loss_sem_bd`
3. Replace dataset root and class settings in LWIR base config.
4. Validate edge map generation pipeline for your labels.
5. Run short overfit test on a tiny subset.

## Suggested Training Command

```bash
python tools/train.py configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py --cfg-options train_cfg.max_iters=1 train_cfg.val_interval=1
```
