# ELSNet: Efficient LWIR Segmentation Network (MMSegmentation-Based)

This repository implements the architecture proposed in:

**Real-Time Long-Wave Infrared Semantic Segmentation With Adaptive Noise Reduction and Feature Fusion**  
(IEEE Access, 2025)

It also includes practical refinements for NNS generation and denoising training.

## 1) Paper At A Glance

The paper targets real-time LWIR semantic segmentation and focuses on:

- adaptive noise reduction before segmentation (SDM)
- boundary-aware feature enhancement (BEM)
- multi-stream feature fusion (MSFM)
- denoising-aware training with auxiliary losses (including iMSE-style formulation)

The overall design follows a PID-style real-time segmentation backbone and adds LWIR-specific denoising/fusion blocks.

## 2) What Is Implemented In This Repo

### Backbone (`ELSNet`)
File: `mmseg/models/backbones/elsnet.py`

- PID-style backbone with detail/semantic/boundary branches.
- `SDM` denoising module in front of the stem.
- `BEM` on D-branch stages.
- Optional `MSFM` for high/low/boundary stream fusion.
- `NNSGenerator` and denoising utilities:
  - `generate_nns`
  - `denoise_with_wave`
  - `raw_wavelet_pack`
  - `forward_from_denoised`
- Optional teacher path for denoising:
  - `use_sdm_teacher`
  - `update_sdm_teacher`
  - `teacher_denoise_with_wave`

### Decode Head (`ELSHead`)
File: `mmseg/models/decode_heads/els_head.py`

- Training-time multi-branch outputs (`p_logit`, `i_logit`, `d_logit`).
- BAS threshold control (`bas_threshold`).
- Decode losses:
  - `loss_sem_p`
  - `loss_sem_i`
  - `loss_bd`
  - `loss_sem_bd`

### Segmentor (`ELSEncoderDecoder`)
File: `mmseg/models/segmentors/els_encoder_decoder.py`

- Extends `EncoderDecoder`.
- Supports both denoising training paths:
  - classical `IMSELoss`
  - soft-contrastive wavelet loss (`SoftContrastiveWaveletLoss`)
- Handles SDM/NNS/teacher wavelet flow for denoising loss orchestration.

### Losses
File: `mmseg/models/losses/lwir_losses.py`

- `IMSELoss` (inverse-MSE option in wavelet domain)
- `SoftContrastiveWaveletLoss` (anchor-positive-negative wavelet contrastive loss)
- `LowSemanticLoss`
- `BoundarySemanticLoss` (`hard` / `soft`)

### Hook
File: `mmseg/engine/hooks/sdm_teacher_ema_hook.py`

- `SDMTeacherEMAHook` for EMA update of SDM teacher during training.

### Registry Wiring

- `mmseg/models/backbones/__init__.py`
- `mmseg/models/decode_heads/__init__.py`
- `mmseg/models/segmentors/__init__.py`
- `mmseg/models/losses/__init__.py`
- `mmseg/engine/hooks/__init__.py`

## 3) What Was Changed / Improved Beyond The Base Paper Flow

This repo is not a strict frozen copy. Main practical changes are:

1. NNS generation refinement  
`NNSGenerator` is implemented as a sensor-like correlated noise process (offset/gain + directional correlation), instead of a simple fixed synthetic pattern.

2. Denoising training refinement  
In addition to `IMSELoss`, this repo supports `SoftContrastiveWaveletLoss` and corresponding wavelet tensor flow in `ELSEncoderDecoder`.

3. Teacher-assisted denoising refinement  
Optional SDM teacher + EMA update hook are provided (`use_sdm_teacher`, `SDMTeacherEMAHook`).

4. Implementation-level stability improvements  
Wavelet-domain packing/unpacking and denoising utility paths are integrated directly in backbone/segmentor for consistent training behavior.

## 4) Active Configs In This Repo

- Train config:
  - `configs/elsnet/elsnet-m_2xb6-120k_640x480-soda.py`
- Test config:
  - `configs/elsnet/elsnet-m_2xb6-120k_640x480-soda_test.py`
- Dataset base:
  - `configs/_base_/datasets/soda_640x480.py`
  - `configs/_base_/datasets/lwir_cityscapes_1024x1024.py` (available as base dataset config)

## 5) Quick Start

### Environment

```bash
pip install -U pip
pip install -r requirements/runtime.txt
pip install -r requirements/mminstall.txt
pip install -e .
```

If required by your runtime path:

```bash
pip install ftfy regex
```

### Train

```bash
python tools/train.py configs/elsnet/elsnet-m_2xb6-120k_640x480-soda.py
```

### Test

```bash
python tools/test.py \
  configs/elsnet/elsnet-m_2xb6-120k_640x480-soda_test.py \
  <CHECKPOINT_PATH>
```

## 6) Expected Training Log Keys

- `loss_imse`
- `decode.loss_sem_p`
- `decode.loss_sem_i`
- `decode.loss_bd`
- `decode.loss_sem_bd`

## 7) Notes

- Dataset paths / class metadata / label mapping should be adjusted for your local setup.
- This README summarizes both paper-aligned structure and current repository-level refinements.
