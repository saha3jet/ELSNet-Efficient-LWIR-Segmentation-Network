# ELSNet: Efficient LWIR Segmentation Network

This repository contains the ELSNet implementation for long-wave infrared semantic segmentation.

Codebase note: this project is built on top of the MMSegmentation framework, but this README focuses on ELSNet-specific implementation and usage.

## Project Scope

- Goal: real-time LWIR semantic segmentation with denoising and boundary-aware fusion.
- Core idea: integrate SDM, BEM, MSFM, and multi-term losses into a PID-style architecture.
- Current naming: ELSNet / ELSHead / ELSEncoderDecoder.

## Current Implementation Status

### Implemented

- Backbone: `mmseg/models/backbones/elsnet.py`
  - SDM before stem.
  - BEM on D-branch stages.
  - Optional MSFM integration.
  - NNS generator and `generate_nns()`.
  - `forward_from_denoised()` for training orchestration.
- Decode head: `mmseg/models/decode_heads/els_head.py`
  - PID-style multi-branch output flow with ELS naming.
  - BAS threshold is configurable via `bas_threshold`.
- Segmentor: `mmseg/models/segmentors/els_encoder_decoder.py`
  - Extends EncoderDecoder and adds `loss_imse` orchestration.
  - Reuses optional wavelet coefficients from backbone when available.
- Losses: `mmseg/models/losses/lwir_losses.py`
  - `IMSELoss` (supports inverse wavelet-domain mode).
  - `IMSELoss` options: `per_channel`, `inverse_transform`, `inverse_clip_max`.
  - `LowSemanticLoss`.
  - `BoundarySemanticLoss` (`hard` and `soft` modes).
- Registry wiring completed:
  - `mmseg/models/backbones/__init__.py`
  - `mmseg/models/decode_heads/__init__.py`
  - `mmseg/models/segmentors/__init__.py`
  - `mmseg/models/losses/__init__.py`
- Configs:
  - Main: `configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py`
  - Dataset base: `configs/_base_/datasets/lwir_cityscapes_1024x1024.py`

### In Progress / Next

- Dataset-finalization for the actual LWIR dataset:
  - `data_root`, class meta, label mapping, edge map validation.
- Runtime verification:
  - 1-iter smoke train (loss keys, finite checks).
  - tiny-subset overfit sanity check.
- Training recipe tuning:
  - lambda balancing for `loss_imse`, `loss_sem_p`, `loss_sem_i`, `loss_bd`, `loss_sem_bd`.

## Architecture Summary

- Input: 1-channel LWIR image.
- Backbone flow:
  - SDM denoises input.
  - PID-style branches process semantic/detail/boundary streams.
  - BEM strengthens boundary stream.
  - MSFM fuses multi-stream information.
- Head:
  - ELSHead outputs `p_logit`, `i_logit`, `d_logit` during training.
- Segmentor-level loss orchestration:
  - ELSEncoderDecoder computes decode losses and additional `loss_imse`.
  - If available, `denoise_with_wave()` is used to avoid recomputing wavelet features for iMSE.

## Loss Composition

- Segmentation/decode side:
  - `decode.loss_sem_p`
  - `decode.loss_sem_i`
  - `decode.loss_bd`
  - `decode.loss_sem_bd`
- Denoising side:
  - `loss_imse`

`IMSELoss` supports inverse wavelet-domain form:

- `L_iMSE = 1 / (MSE(W_d, W_n) + eps)`

where `W_d` and `W_n` are Haar-wavelet coefficients of denoised and NNS samples.

Current default config uses OHEM for BAS-equivalent supervision in the 4th decode loss slot (same pattern as PID config family).

## Quick Start

### 1) Environment

Install required runtime packages first (`torch`, `mmengine`, `mmcv`, and project requirements).

### 2) Smoke Train (1 Iteration)

```bash
python tools/train.py configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py --cfg-options train_cfg.max_iters=1 train_cfg.val_interval=1
```

### 3) Validate Loss Keys

Check logs for:

- `loss_imse`
- `decode.loss_sem_p`
- `decode.loss_sem_i`
- `decode.loss_bd`
- `decode.loss_sem_bd`

## Repository Pointers

- ELSNet backbone: `mmseg/models/backbones/elsnet.py`
- ELSHead: `mmseg/models/decode_heads/els_head.py`
- ELS segmentor: `mmseg/models/segmentors/els_encoder_decoder.py`
- LWIR losses: `mmseg/models/losses/lwir_losses.py`
- Main config: `configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py`
- Dataset base config: `configs/_base_/datasets/lwir_cityscapes_1024x1024.py`
- Additional summary: `docs/ELSNET_REPO_SUMMARY.md`

## Notes

- This README intentionally excludes generic MMSegmentation model-zoo/tutorial content.
- For framework internals, use upstream MMSegmentation docs as reference.
