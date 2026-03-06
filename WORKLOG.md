# Worklog

## Scope
This file tracks implementation progress for variation-aware proxy segmentation on top of MMSegmentation.

## Completed

1. Repository setup
- Cloned MMSegmentation into a clean project repo.
- Detached from upstream and connected to private repository.

2. Proxy loss implementation
- Added `VariationAwareProxyLoss`:
  - file: `mmseg/models/losses/variation_aware_proxy_loss.py`
  - components:
    - representative proxy loss (`loss_proxy_repr`)
    - variation matching loss (`loss_proxy_var`)
    - optional diversity regularization (`loss_proxy_diversity`)

3. Proxy-enabled decoder heads
- Added `proxy_heads.py`:
  - file: `mmseg/models/decode_heads/proxy_heads.py`
  - implemented heads:
    - `FCNProxyHead`
    - `PSPProxyHead`
    - `ASPPProxyHead`
    - `DepthwiseSeparableASPPProxyHead`
    - `LRASPPProxyHead`
    - `SegformerProxyHead`
    - `LightHamProxyHead`
    - `PIDProxyHead`
- Pixel-wise embedding pipeline integrated via mixin.
- Added `proxy_upsample_first` option for resolution strategy ablation.

4. Registry wiring
- Updated:
  - `mmseg/models/decode_heads/__init__.py`
  - `mmseg/models/losses/__init__.py`

5. Config templates for reproduction
- Added configs under `configs/varp/` for:
  - HRNet
  - MobileNetV2 (FCN / PSPNet / DeepLabV3 / DeepLabV3+)
  - MobileNetV3 (LRASPP)
  - PIDNet
  - SegNeXt
  - SegFormer

6. Sanity check
- Performed Python compile checks for added code/config files.

## In Progress

1. Documentation polish
- Refine README wording/examples for easier onboarding.
- Add quick-start train commands per backbone.

## Planned

1. Experiment protocol finalization
- Define canonical train/eval scripts per backbone.
- Add recommended hyperparameter defaults per dataset.

2. Reproducibility artifacts
- Add result table template and run logs directory convention.
- Add minimal checks for loss term scale monitoring.

3. Optional engineering tasks
- Add unit tests for proxy loss shape/ignore-index behavior.
- Add ablation configs for `proxy_upsample_first=True`.

## Update (Paper Sync)

- Located and parsed local paper PDF:
  - `(local file path removed for public repository)`
- Reflected paper-level definitions in README:
  - factorized similarity score,
  - negative-only focal modulation,
  - compositional similarity loss (`L_a`, `L_r`, `L_cs`),
  - default paper hyperparameters (`lambda_var`, `K_c`, `tau`, `gamma`, `tau_R`, `lambda_r`).

## Change Log (Key Files)

- `README.md`
- `WORKLOG.md`
- `mmseg/models/losses/variation_aware_proxy_loss.py`
- `mmseg/models/decode_heads/proxy_heads.py`
- `mmseg/models/losses/__init__.py`
- `mmseg/models/decode_heads/__init__.py`
- `configs/varp/**`
