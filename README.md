# ELSNet: Efficient LWIR Segmentation Network (MMSegmentation-Based)

This repository contains an ELSNet implementation for long-wave infrared (LWIR) semantic segmentation, built on top of MMSegmentation.

## Implemented Components

### 1) Backbone: `ELSNet`
File: `mmseg/models/backbones/elsnet.py`

- PID-style backbone structure with detail/semantic/boundary streams.
- `SDM` (denoising module) before stem.
- `BEM` (boundary enhancement module) on D-branch stages.
- Optional `MSFM` (multi-stream feature fusion module).
- `NNSGenerator` for negative noise sample generation.
- Wavelet utility paths for denoising training:
  - `denoise_with_wave`
  - `raw_wavelet_pack`
  - `forward_from_denoised`
- Optional SDM teacher (EMA):
  - `use_sdm_teacher`
  - `update_sdm_teacher`
  - `teacher_denoise_with_wave`

### 2) Decode Head: `ELSHead`
File: `mmseg/models/decode_heads/els_head.py`

- Multi-branch training outputs:
  - `p_logit` (P-branch)
  - `i_logit` (I-branch)
  - `d_logit` (boundary)
- BAS threshold control via `bas_threshold`.
- Computes decode losses:
  - `loss_sem_p`
  - `loss_sem_i`
  - `loss_bd`
  - `loss_sem_bd`

### 3) Segmentor: `ELSEncoderDecoder`
File: `mmseg/models/segmentors/els_encoder_decoder.py`

- Extends `EncoderDecoder`.
- Supports two denoising loss paths:
  - Classic `IMSELoss` path.
  - Soft-contrastive wavelet path (`SoftContrastiveWaveletLoss`, Option 5-C).
- Uses SDM/NNS/teacher wavelet tensors for contrastive denoising when configured.

### 4) Losses
File: `mmseg/models/losses/lwir_losses.py`

- `IMSELoss`
  - Inverse-MSE option in wavelet domain.
  - Wavelet packing aligned with SDM.
- `SoftContrastiveWaveletLoss`
  - Anchor/positive/negative wavelet contrastive formulation.
  - HF-only option (drop LL).
- `LowSemanticLoss`
- `BoundarySemanticLoss` (`hard` / `soft`)

### 5) Hook
File: `mmseg/engine/hooks/sdm_teacher_ema_hook.py`

- `SDMTeacherEMAHook` updates SDM teacher by EMA after train iterations.

### 6) Registry Integration

- Backbones: `mmseg/models/backbones/__init__.py`
- Decode heads: `mmseg/models/decode_heads/__init__.py`
- Segmentors: `mmseg/models/segmentors/__init__.py`
- Losses: `mmseg/models/losses/__init__.py`
- Hooks: `mmseg/engine/hooks/__init__.py`

## Available ELSNet Configs

- `configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py`
  - Cityscapes-style LWIR setup
  - `IMSELoss` path
- `configs/elsnet/elsnet-m_2xb6-80k_640x480-soda.py`
  - SODA setup
  - `SoftContrastiveWaveletLoss` + `SDMTeacherEMAHook`

## Dataset Base Configs

- `configs/_base_/datasets/lwir_cityscapes_1024x1024.py`
- `configs/_base_/datasets/soda_640x480.py`

## Quick Start

### 1) Environment (example)

Install PyTorch, MMCV, MMEngine, then install this repo:

```bash
pip install -U pip
pip install -r requirements/runtime.txt
pip install -r requirements/mminstall.txt
pip install -e .
```

If needed by your pipeline/modules:

```bash
pip install ftfy regex
```

### 2) Train

```bash
python tools/train.py configs/elsnet/elsnet-m_2xb6-80k_640x480-soda.py
```

or

```bash
python tools/train.py configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py
```

### 3) Validate Key Loss Terms

Expected logs include:

- `loss_imse`
- `decode.loss_sem_p`
- `decode.loss_sem_i`
- `decode.loss_bd`
- `decode.loss_sem_bd`

## Current Notes

- This repository focuses on ELSNet-specific implementation details.
- Dataset paths, class metadata, and label mapping should be adjusted to your local dataset.
- `configs/elsnet/elsnet-s_2xb6-120k_1024x1024-cityscapes.py` currently references `class_weight` in `loss_decode`; ensure `class_weight` is defined or set to `None` before training.
