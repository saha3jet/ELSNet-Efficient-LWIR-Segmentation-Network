# Variation-Aware Proxy Segmentation

This repository reproduces and extends the paper:

- **Variation-aware proxy learning for semantic segmentation**  
  (Neurocomputing 659, 2026, Article 131783)

## Installation

Installation is the same as MMSegmentation.

- Please follow: `docs/en/get_started.md#installation`
- Dataset preparation guide: `docs/en/user_guides/2_dataset_prepare.md#prepare-datasets`

## Theoretical Summary (from Paper)

### 1. Motivation

Single-proxy-per-class methods improve inter-class separation, but they are weak at modeling **intra-class variation** (different appearances within one class). This issue is severe near semantic boundaries and in complex scenes.

### 2. Representation Design

For each class `c`, the method learns:

- a **representative proxy** `P_c` (shared class semantics),
- multiple **variation vectors** `v_{c,i}` (fine-grained intra-class modes).

All embeddings/proxies/vectors are L2-normalized.

### 3. Factorized Similarity Score

For a pixel embedding `x`, class `c`, variation index `i`:

`s(x, v_{c,i}) = sim(x, P_c) + lambda_var * sim(x, v_{c,i})`

- `sim` is cosine similarity.
- Default in paper: `lambda_var = 1.0`, `K_c = 5` variation vectors/class.

This factorization combines global class semantics and local class variation in one score.

### 4. Focal Modulation (Negative-only)

The paper applies focal modulation only on hard negatives:

- `p_sub(x, v_{c,i}) = softmax(tau * s(x, v_{c,i}))`
- `p_neg` is max probability over non-GT class variations.
- modulation factor: `M_r = (p_neg)^gamma`

Defaults in paper: `tau = 10.0`, `gamma = 2.0`, hard-negative threshold `tau_R = 0.8`.

### 5. Compositional Similarity Loss

The objective has two parts:

- **Attraction loss (`L_a`)**: pulls embeddings toward GT-class variations.
- **Repulsion loss (`L_r`)**: pushes hard negatives away using `M_r`.

Combined form:

- `L_cs = L_a + lambda_r * L_r`, with default `lambda_r = 1.0`.
- Final training adds this as auxiliary term to segmentation loss (`lambda_cs` default 1.0 in paper setup).

### 6. Training vs Inference

- The proxy branch is used during **training** to shape embedding space.
- At **inference**, the base encoder-decoder path is used, so no extra inference-time overhead from the auxiliary proxy-learning branch.

## Implemented in This Repo

### New Loss

- `mmseg/models/losses/variation_aware_proxy_loss.py`

### New Decode Heads

- `mmseg/models/decode_heads/proxy_heads.py`
- supported heads:
  - `FCNProxyHead`
  - `PSPProxyHead`
  - `ASPPProxyHead`
  - `DepthwiseSeparableASPPProxyHead`
  - `LRASPPProxyHead`
  - `SegformerProxyHead`
  - `LightHamProxyHead`
  - `PIDProxyHead`

### Registry Updates

- `mmseg/models/decode_heads/__init__.py`
- `mmseg/models/losses/__init__.py`

### Added Reproduction Configs

- `configs/varp/hrnet/fcn_hr18_4xb2-40k_cityscapes-512x1024_proxy.py`
- `configs/varp/mobilenet_v2/mobilenet-v2-d8_fcn_4xb2-80k_cityscapes-512x1024_proxy.py`
- `configs/varp/mobilenet_v2/mobilenet-v2-d8_pspnet_4xb2-80k_cityscapes-512x1024_proxy.py`
- `configs/varp/mobilenet_v2/mobilenet-v2-d8_deeplabv3_4xb2-80k_cityscapes-512x1024_proxy.py`
- `configs/varp/mobilenet_v2/mobilenet-v2-d8_deeplabv3plus_4xb2-80k_cityscapes-512x1024_proxy.py`
- `configs/varp/mobilenet_v3/mobilenet-v3-d8_lraspp_4xb4-320k_cityscapes-512x1024_proxy.py`
- `configs/varp/pidnet/pidnet-s_2xb6-120k_1024x1024-cityscapes_proxy.py`
- `configs/varp/segnext/segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512_proxy.py`
- `configs/varp/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024_proxy.py`

## Notes

- This project preserves attribution and licensing obligations of MMSegmentation and the original paper.
- The current implementation follows the same core principle: **pixel-wise latent supervision** for multi-class semantic segmentation.
