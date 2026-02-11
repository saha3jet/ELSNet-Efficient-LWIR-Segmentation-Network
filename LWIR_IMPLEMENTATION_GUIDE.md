# LWIR Semantic Segmentation Implementation Guide

**Based on**: "Real-Time Long-Wave Infrared Semantic Segmentation With Adaptive Noise Reduction and Feature Fusion" (IEEE Access, 2025)

**Framework**: mmsegmentation + PIDNet

**Status**: Phase 1 (Utilities) COMPLETED ✅ | Phase 2 (Integration) IN PROGRESS 🚧

---

## ✅ Completed Files

### 1. Wavelet Utilities
**File**: `mmseg/models/utils/wavelet.py`
- ✅ `haar_dwt(x)`: Decomposes (B,C,H,W) → (B,C*4,H/2,W/2)
- ✅ `haar_idwt(coeffs)`: Reconstructs (B,C*4,H/2,W/2) → (B,C,H,W)
- ✅ Gradient-friendly, numerically stable
- ✅ Round-trip tested: IDWT(DWT(x)) ≈ x

### 2. ECA Layer
**File**: `mmseg/models/utils/eca_layer.py`
- ✅ Efficient Channel Attention with adaptive kernel size
- ✅ Formula: k = |log2(C) + b| / gamma (ensures odd k)
- ✅ 1D convolution over channel dimension

### 3. AWM (Adaptive Wavelet Module)
**File**: `mmseg/models/utils/awm.py`
- ✅ Learnable weights for 4 subbands: α(LL), β(LH), γ(HL), ω(HH)
- ✅ **Initialization** (ALTERNATIVE - as requested):
  - `alpha = 1.0` (preserve low-frequency)
  - `beta = gamma = omega = 0.5` (reduce high-frequency noise)
- ✅ Commented version with equal weighting (1.0 for all) included

### 4. SDM (Stripe Denoising Module)
**File**: `mmseg/models/utils/sdm.py`
- ✅ Input: 1-channel grayscale LWIR (B, 1, H, W)
- ✅ Pipeline: DWT → Conv Stack → IDWT
- ✅ Registered with `@MODELS.register_module()`
- ✅ Configurable: `in_channels=1`, `mid_channels=64`, `num_layers=3`

---

## 🚧 Remaining Implementation Tasks

### Phase 2A: Core Modules (Remaining)

#### 5. BEM (Boundary Enhancement Module)
**File**: `mmseg/models/utils/bem.py`

```python
# Key Components:
# 1. Channel pooling (max + avg) → (B, 1, H, W) each
# 2. AWM applied to both pooled features
# 3. Upsample + concatenate → (B, 2, H, W)
# 4. Spatial attention via Conv(2→1) + Sigmoid
# 5. Apply attention: f_b * attention + f_b (residual)
# 6. ECA for channel attention

@MODELS.register_module()
class BEM(BaseModule):
    def __init__(self, in_channels, norm_cfg, act_cfg, init_cfg):
        self.awm_max = AWM()
        self.awm_avg = AWM()
        self.spatial_conv = ConvModule(2, 1, kernel_size=7, padding=3)
        self.eca = ECALayer(in_channels)
    
    def forward(self, f_b):
        # Max/avg pooling across channels
        max_pool = torch.max(f_b, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(f_b, dim=1, keepdim=True)
        
        # AWM processing (output at half resolution)
        max_w = self.awm_max(max_pool)
        avg_w = self.awm_avg(avg_pool)
        
        # Upsample back to original resolution
        max_up = F.interpolate(max_w, size=f_b.shape[2:], mode='bilinear')
        avg_up = F.interpolate(avg_w, size=f_b.shape[2:], mode='bilinear')
        
        # Spatial attention
        spatial_input = torch.cat([max_up, avg_up], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_input))
        
        # Apply attention + residual
        f_b_attn = f_b * spatial_attn
        f_b_res = f_b + f_b_attn
        
        # Channel attention
        return self.eca(f_b_res)
```

#### 6. MSFM (Multi-Stream Fusion Module)
**File**: `mmseg/models/utils/msfm.py`

```python
# Key Components:
# 1. Local enhancement: axis pooling (vertical/horizontal) attention
# 2. Downsample enhanced features → gate for high semantic
# 3. Global enhancement: PAG module (reuse from PIDNet)
# 4. Final fusion: concat all streams

@MODELS.register_module()
class MSFM(BaseModule):
    def __init__(self, hs_channels=256, ls_channels=128, b_channels=128, ...):
        # Axis pooling convs
        self.v_conv = ConvModule(ls_channels + b_channels, ls_channels, 1)
        self.h_conv = ConvModule(ls_channels + b_channels, ls_channels, 1)
        
        # Downsample for gating
        self.downsample = ConvModule(ls_channels * 2, hs_channels, 3, stride=2, padding=1)
        
        # PAG for global enhancement (from PIDNet)
        self.pag = PagFM(hs_channels, hs_channels // 2)
        
        # Final fusion
        self.fusion_conv = ConvModule(hs_channels + ls_channels + b_channels, out_channels, 3, padding=1)
    
    def forward(self, f_hs, f_ls, f_b):
        # Local: axis pooling attention
        f_comb = torch.cat([f_ls, f_b], dim=1)
        
        # Vertical pooling (pool over W)
        v_pool = torch.mean(f_comb, dim=3, keepdim=True).expand_as(f_comb)
        f_v_attn = torch.sigmoid(self.v_conv(v_pool))
        
        # Horizontal pooling (pool over H)
        h_pool = torch.mean(f_comb, dim=2, keepdim=True).expand_as(f_comb)
        f_h_attn = torch.sigmoid(self.h_conv(h_pool))
        
        # Apply to low semantic
        f_v = f_ls * f_v_attn[:, :f_ls.shape[1]]
        f_h = f_ls * f_h_attn[:, :f_ls.shape[1]]
        f_els = torch.cat([f_v, f_h], dim=1)
        
        # Gate high semantic
        gate = torch.sigmoid(self.downsample(f_els))
        gate = F.interpolate(gate, size=f_hs.shape[2:], mode='bilinear')
        f_hs_gated = f_hs * gate
        
        # Global: PAG fusion
        f_hs_up = F.interpolate(f_hs_gated, size=f_ls.shape[2:], mode='bilinear')
        f_global = self.pag(f_ls, f_hs_up)
        
        # Enhance boundary with global context
        f_b_enhanced = f_b + F.interpolate(f_hs_up, size=f_b.shape[2:], mode='bilinear')
        
        # Final fusion
        fused = torch.cat([f_global, f_hs_up, f_b_enhanced], dim=1)
        return self.fusion_conv(fused)
```

---

### Phase 2B: PIDNet Integration

#### 7. PIDNetLWIR Backbone
**File**: `mmseg/models/backbones/pidnet_lwir.py`

**Strategy**: Copy `PIDNet` class → Add SDM + BEM + MSFM

**Key Integration Points**:

```python
@MODELS.register_module()
class PIDNetLWIR(BaseModule):
    """PIDNet backbone with LWIR enhancements.
    
    Modifications from original PIDNet:
    1. SDM applied before stem layer
    2. BEM applied after each D-branch stage (3x total)
    3. MSFM applied at 2 fusion points
    4. Returns enhanced features for LWIR decode head
    """
    
    def __init__(self, in_channels=1, channels=64, ...):
        super().__init__()
        
        # [NEW] SDM for stripe denoising
        self.sdm = SDM(in_channels=in_channels)
        
        # Original PIDNet components
        self.stem = self._make_stem_layer(...)
        self.i_branch_layers = nn.ModuleList([...])
        self.p_branch_layers = nn.ModuleList([...])
        self.d_branch_layers = nn.ModuleList([...])
        
        # [NEW] BEM modules for boundary enhancement
        self.bem_1 = BEM(in_channels=channels)  # After D-branch stage 1
        self.bem_2 = BEM(in_channels=channels)  # After D-branch stage 2
        self.bem_3 = BEM(in_channels=channels * 2)  # After D-branch stage 3
        
        # [NEW] MSFM modules
        self.msfm_1 = MSFM(hs_channels=256, ls_channels=128, b_channels=64)
        self.msfm_2 = MSFM(hs_channels=256, ls_channels=128, b_channels=128)
        
        # Original fusion modules
        self.pag_1 = PagFM(...)
        self.pag_2 = PagFM(...)
        self.spp = PAPPM(...)
        self.dfm = LightBag(...)  # or Bag depending on num_stem_blocks
    
    def forward(self, x):
        # [NEW] Apply SDM before stem
        x = self.sdm(x)  # (B, 1, H, W) → (B, 1, H, W) denoised
        
        # Original stem
        x = self.stem(x)  # (B, 64, H/4, W/4) → (B, 128, H/8, W/8)
        
        # === Stage 3 ===
        x_i = self.relu(self.i_branch_layers[0](x))  # (B, 128, H/8, W/8)
        x_p = self.p_branch_layers[0](x)              # (B, 128, H/8, W/8)
        x_d = self.d_branch_layers[0](x)              # (B, 64, H/8, W/8)
        
        # [NEW] BEM for boundary stream
        x_d = self.bem_1(x_d)
        
        # Original PAG fusion
        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(diff_i, size=x_d.shape[2:], ...)
        
        if self.training:
            temp_p = x_p.clone()
        
        # === Stage 4 ===
        x_i = self.relu(self.i_branch_layers[1](x_i))  # (B, 256, H/16, W/16)
        x_p = self.p_branch_layers[1](self.relu(x_p))  # (B, 128, H/8, W/8)
        x_d = self.d_branch_layers[1](self.relu(x_d))  # (B, 64, H/8, W/8)
        
        # [NEW] BEM
        x_d = self.bem_2(x_d)
        
        # Original fusion
        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(diff_i, size=x_d.shape[2:], ...)
        
        if self.training:
            temp_d = x_d.clone()
        
        # === Stage 5 ===
        x_i = self.i_branch_layers[2](x_i)             # (B, 512, H/32, W/32)
        x_p = self.p_branch_layers[2](self.relu(x_p)) # (B, 128, H/8, W/8)
        x_d = self.d_branch_layers[2](self.relu(x_d)) # (B, 128, H/8, W/8)
        
        # [NEW] BEM
        x_d = self.bem_3(x_d)
        
        # SPP + DFM (original)
        x_i = self.spp(x_i)  # (B, 256, H/8, W/8)
        x_i = F.interpolate(x_i, size=x_p.shape[2:], ...)
        out = self.dfm(x_p, x_i, x_d)  # (B, 256, H/8, W/8)
        
        # Return multi-stream outputs
        if self.training:
            return (temp_p, out, temp_d)
        else:
            return out
```

**Registration**:
```python
# In mmseg/models/backbones/__init__.py
from .pidnet_lwir import PIDNetLWIR

__all__ = [..., 'PIDNetLWIR']
```

---

### Phase 3: Custom Losses

#### 8. LWIR Losses
**File**: `mmseg/models/losses/lwir_losses.py`

```python
@MODELS.register_module()
class IMSELoss(nn.Module):
    """Illumination-weighted MSE Loss (L_iMSE).
    
    Loss weight is configurable via config file.
    """
    def __init__(self, loss_weight=1.0, loss_name='loss_imse'):
        super().__init__()
        self.loss_weight = loss_weight  # ← CONFIGURABLE
        self.loss_name_ = loss_name
    
    def forward(self, x_d, x):
        # MSE between denoised and original
        loss = F.mse_loss(x_d, x, reduction='mean')
        return self.loss_weight * loss
    
    @property
    def loss_name(self):
        return self.loss_name_


@MODELS.register_module()
class LowSemanticLoss(nn.Module):
    """Low-level semantic loss (L_ls)."""
    def __init__(self, loss_weight=1.0, loss_name='loss_ls', ignore_index=255):
        super().__init__()
        self.loss_weight = loss_weight  # ← CONFIGURABLE
        self.loss_name_ = loss_name
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, seg_logit, seg_label):
        loss = self.ce_loss(seg_logit, seg_label)
        return self.loss_weight * loss
    
    @property
    def loss_name(self):
        return self.loss_name_


@MODELS.register_module()
class BoundarySemanticLoss(nn.Module):
    """Boundary-aware semantic loss (L_bas).
    
    Only computes semantic loss where boundary confidence > threshold.
    """
    def __init__(self, threshold=0.8, loss_weight=1.0, 
                 loss_name='loss_bas', ignore_index=255):
        super().__init__()
        self.threshold = threshold
        self.loss_weight = loss_weight  # ← CONFIGURABLE
        self.loss_name_ = loss_name
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, seg_logit, seg_label, bd_pred):
        # Mask semantic labels where boundary prob > threshold
        bd_prob = torch.sigmoid(bd_pred[:, 0, :, :])
        filler = torch.ones_like(seg_label) * 255  # ignore_index
        masked_label = torch.where(bd_prob > self.threshold, seg_label, filler)
        
        loss = self.ce_loss(seg_logit, masked_label)
        return self.loss_weight * loss
    
    @property
    def loss_name(self):
        return self.loss_name_
```

**Config Usage**:
```python
# In config file (*.py)
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='PIDNetLWIR', ...),
    decode_head=dict(
        type='LWIRPIDHead',
        loss_decode=[
            dict(type='LowSemanticLoss', loss_weight=1.0),      # L_ls
            dict(type='CrossEntropyLoss', loss_weight=1.0),     # L_seg
            dict(type='BoundaryLoss', loss_weight=1.0),         # L_bd
            dict(type='BoundarySemanticLoss', loss_weight=1.0), # L_bas
            dict(type='IMSELoss', loss_weight=0.1),             # L_iMSE
        ]
    )
)
```

**Registration**:
```python
# In mmseg/models/losses/__init__.py
from .lwir_losses import IMSELoss, LowSemanticLoss, BoundarySemanticLoss

__all__ = [..., 'IMSELoss', 'LowSemanticLoss', 'BoundarySemanticLoss']
```

---

### Phase 4: Decode Head

#### 9. LWIR PID Head
**File**: `mmseg/models/decode_heads/lwir_pid_head.py`

```python
@MODELS.register_module()
class LWIRPIDHead(BaseDecodeHead):
    """Decode head for PIDNetLWIR.
    
    Handles multi-stream outputs (P/I/D branches) and computes
    multiple losses (semantic, low-semantic, boundary, boundary-aware).
    """
    
    def __init__(self, in_channels, channels, num_classes, ...):
        super().__init__(in_channels, channels, num_classes, ...)
        
        # Three parallel heads (from PIDHead)
        self.i_head = BasePIDHead(in_channels, channels)
        self.p_head = BasePIDHead(in_channels // 2, channels)
        self.d_head = BasePIDHead(in_channels // 2, in_channels // 4)
        
        # Classification layers
        self.p_cls_seg = nn.Conv2d(channels, num_classes, 1)  # Low-semantic
        self.d_cls_seg = nn.Conv2d(in_channels // 4, 1, 1)    # Boundary (binary)
    
    def forward(self, inputs):
        if self.training:
            x_p, x_i, x_d = inputs  # Multi-stream from backbone
            
            # Generate logits
            p_logit = self.p_head(x_p, self.p_cls_seg)  # Low-semantic
            i_logit = self.i_head(x_i, self.cls_seg)    # Main semantic
            d_logit = self.d_head(x_d, self.d_cls_seg)  # Boundary
            
            return p_logit, i_logit, d_logit
        else:
            # Inference: only use I-branch (main semantic)
            return self.i_head(inputs, self.cls_seg)
    
    def loss_by_feat(self, seg_logits, batch_data_samples):
        """Compute all losses."""
        loss = dict()
        p_logit, i_logit, d_logit = seg_logits
        
        # Extract GT labels
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        
        # Resize logits to match GT resolution
        p_logit = resize(p_logit, size=sem_label.shape[2:], mode='bilinear')
        i_logit = resize(i_logit, size=sem_label.shape[2:], mode='bilinear')
        d_logit = resize(d_logit, size=bd_label.shape[2:], mode='bilinear')
        
        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)
        
        # Compute losses via loss_decode list
        loss['loss_sem_p'] = self.loss_decode[0](p_logit, sem_label)  # L_ls
        loss['loss_sem_i'] = self.loss_decode[1](i_logit, sem_label)  # L_seg
        loss['loss_bd'] = self.loss_decode[2](d_logit, bd_label)      # L_bd
        loss['loss_sem_bd'] = self.loss_decode[3](i_logit, sem_label, d_logit)  # L_bas
        
        # [TODO] Add L_iMSE if SDM outputs are available
        # loss['loss_imse'] = self.loss_decode[4](x_denoised, x_original)
        
        # Accuracy
        loss['acc_seg'] = accuracy(i_logit, sem_label, ignore_index=self.ignore_index)
        
        return loss
    
    def _stack_batch_gt(self, batch_data_samples):
        """Extract semantic and boundary GT from batch."""
        gt_semantic_segs = [s.gt_sem_seg.data for s in batch_data_samples]
        gt_edge_segs = [s.gt_edge_map.data for s in batch_data_samples]
        
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        
        return gt_sem_segs, gt_edge_segs
```

---

### Phase 5: NNS & Boundary GT Generation

#### 10. NNS (Negative Noise Sample) Generation
**File**: `mmseg/models/utils/nns.py`

```python
def generate_stripe_noise(x, lambda_noise=0.2, stripe_direction='both'):
    """Generate stripe noise for LWIR images.
    
    Args:
        x (Tensor): Input LWIR image (B, 1, H, W).
        lambda_noise (float): Noise mixing ratio. Default: 0.2.
        stripe_direction (str): 'horizontal', 'vertical', or 'both'. Default: 'both'.
    
    Returns:
        Tensor: Noisy sample x_n with enhanced stripe artifacts.
    """
    B, C, H, W = x.shape
    device = x.device
    
    noise = torch.zeros_like(x)
    
    if stripe_direction in ['horizontal', 'both']:
        # Horizontal stripes (row-wise artifacts)
        # Random intensity per row
        row_noise = torch.randn(B, C, H, 1, device=device) * 0.1
        row_noise = row_noise.expand(-1, -1, -1, W)
        noise += row_noise
    
    if stripe_direction in ['vertical', 'both']:
        # Vertical stripes (column-wise artifacts)
        col_noise = torch.randn(B, C, 1, W, device=device) * 0.1
        col_noise = col_noise.expand(-1, -1, H, -1)
        noise += col_noise
    
    # Mix with original image
    x_n = x + lambda_noise * noise
    
    # Clamp to valid range
    x_n = torch.clamp(x_n, x.min(), x.max())
    
    return x_n


def compute_l_imse_loss(x_d_wavelet, x_n_wavelet, eps=1e-6):
    """Compute inverse MSE loss in wavelet domain.
    
    Args:
        x_d_wavelet (Tensor): Denoised wavelet coefficients (B, C*4, H/2, W/2).
        x_n_wavelet (Tensor): Noisy wavelet coefficients (B, C*4, H/2, W/2).
        eps (float): Stability constant. Default: 1e-6.
    
    Returns:
        Tensor: Inverse MSE loss (scalar).
    """
    mse = F.mse_loss(x_d_wavelet, x_n_wavelet, reduction='mean')
    return 1.0 / (mse + eps)
```

#### 11. Boundary GT Generation
**File**: `mmseg/datasets/pipelines/generate_boundary.py`

```python
import cv2
import numpy as np

def generate_boundary_gt(seg_label, thickness=3, ignore_index=255):
    """Generate boundary ground truth from semantic segmentation labels.
    
    Args:
        seg_label (ndarray): Semantic label map (H, W).
        thickness (int): Boundary thickness in pixels. Default: 3.
        ignore_index (int): Ignore index value. Default: 255.
    
    Returns:
        ndarray: Binary boundary map (H, W), dtype=uint8.
    """
    # Mask out ignore regions
    mask = (seg_label != ignore_index).astype(np.uint8)
    
    # Compute gradients (morphological gradient)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(seg_label, kernel)
    eroded = cv2.erode(seg_label, kernel)
    
    # Boundary: pixels where dilation != erosion
    boundary = ((dilated != eroded) & (mask == 1)).astype(np.uint8)
    
    # Apply thickness
    if thickness > 1:
        kernel_thick = cv2.getStructuringElement(
            cv2.MORPH_RECT, (thickness, thickness))
        boundary = cv2.dilate(boundary, kernel_thick)
    
    return boundary


# Usage in dataset pipeline:
@TRANSFORMS.register_module()
class GenerateBoundary:
    """Transform to generate boundary GT."""
    
    def __init__(self, thickness=3, ignore_index=255):
        self.thickness = thickness
        self.ignore_index = ignore_index
    
    def __call__(self, results):
        seg_label = results['gt_seg_map']
        boundary = generate_boundary_gt(seg_label, self.thickness, self.ignore_index)
        results['gt_edge_map'] = boundary
        return results
```

---

## 📝 Configuration Example

### Example Config: `configs/pidnet_lwir/pidnet_s_lwir_512x512_80k_lwir_dataset.py`

```python
_base_ = [
    '../_base_/models/pidnet.py',
    '../_base_/datasets/your_lwir_dataset.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# Model
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[128.0],  # Single channel normalization
        std=[64.0],
        bgr_to_rgb=False,
        size=(512, 512)
    ),
    backbone=dict(
        type='PIDNetLWIR',
        in_channels=1,  # Grayscale LWIR
        channels=64,
        ppm_channels=96,
        num_stem_blocks=2,
        align_corners=False,
        # SDM config
        sdm_cfg=dict(
            in_channels=1,
            mid_channels=64,
            num_layers=3
        )
    ),
    decode_head=dict(
        type='LWIRPIDHead',
        in_channels=256,
        channels=128,
        num_classes=19,  # Adjust to your dataset
        align_corners=False,
        loss_decode=[
            dict(type='LowSemanticLoss', loss_weight=1.0),       # L_ls
            dict(type='CrossEntropyLoss', loss_weight=1.0),      # L_seg
            dict(type='BoundaryLoss', loss_weight=1.0),          # L_bd
            dict(type='BoundarySemanticLoss', 
                 loss_weight=1.0, threshold=0.8),                # L_bas
            dict(type='IMSELoss', loss_weight=0.1)               # L_iMSE
        ]
    ),
    # Training cfg
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Data pipeline
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='GenerateBoundary', thickness=3),  # Generate boundary GT
    dict(type='Resize', scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
)

# Learning rate schedule
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=1e-4, by_epoch=False)
]

# Runtime
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=8000)
)
```

---

## 🚀 Quick Start Commands

### Training
```bash
cd <your-project-path>

# Single GPU
python tools/train.py configs/pidnet_lwir/pidnet_s_lwir_512x512_80k_lwir_dataset.py

# Multi-GPU (4 GPUs)
bash tools/dist_train.sh configs/pidnet_lwir/pidnet_s_lwir_512x512_80k_lwir_dataset.py 4
```

### Evaluation
```bash
python tools/test.py \
    configs/pidnet_lwir/pidnet_s_lwir_512x512_80k_lwir_dataset.py \
    checkpoints/pidnet_lwir_iter_80000.pth \
    --eval mIoU
```

### Inference
```bash
python tools/predict.py \
    configs/pidnet_lwir/pidnet_s_lwir_512x512_80k_lwir_dataset.py \
    checkpoints/pidnet_lwir_iter_80000.pth \
    --img path/to/lwir/image.png \
    --out output.png
```

---

## ✅ Definition of Done (DoD)

### Module-Level Tests
- [ ] Wavelet round-trip: `IDWT(DWT(x))` ≈ `x` (MSE < 1e-5)
- [ ] SDM forward: (B,1,256,256) → (B,1,256,256) with finite values
- [ ] BEM forward: (B,128,64,64) → (B,128,64,64) with gradients
- [ ] MSFM forward: 3 inputs → single output, all shapes match
- [ ] AWM parameters: Check `alpha`, `beta`, `gamma`, `omega` in optimizer

### Integration Tests
- [ ] PIDNetLWIR forward (train mode): Returns tuple(temp_p, out, temp_d)
- [ ] PIDNetLWIR forward (eval mode): Returns single tensor
- [ ] All 5 losses return finite scalars
- [ ] Loss backward: No NaN/Inf gradients

### System Tests
- [ ] Overfit on 10-image subset: Loss decreases over 100 iterations
- [ ] Full training run: Completes 1000 iterations without crash
- [ ] Evaluation: mIoU computed successfully
- [ ] Config instantiation: `build_segmentor(cfg.model)` succeeds

---

## 📋 Next Steps

### Priority 1: Complete Core Implementation
1. **BEM module** → `mmseg/models/utils/bem.py`
2. **MSFM module** → `mmseg/models/utils/msfm.py`
3. **Update `__init__.py`** → Register all new modules
4. **Run unit tests** → Test each module independently

### Priority 2: Backbone Integration
5. **PIDNetLWIR** → Copy PIDNet + integrate SDM/BEM/MSFM
6. **Register backbone** → Add to `mmseg/models/backbones/__init__.py`
7. **Test forward pass** → Dummy input (B,1,512,512)

### Priority 3: Losses & Head
8. **Implement losses** → `mmseg/models/losses/lwir_losses.py`
9. **LWIRPIDHead** → `mmseg/models/decode_heads/lwir_pid_head.py`
10. **Register all** → Update `__init__.py` files

### Priority 4: Data Pipeline & Config
11. **Boundary GT generation** → Transform in dataset pipeline
12. **NNS generation** → Integrate in training loop
13. **Create config files** → At least 1 working config
14. **Test training** → 10-image overfit

---

## 🔍 Troubleshooting

### Common Issues

**Issue 1**: `ModuleNotFoundError: No module named 'mmseg.models.utils.wavelet'`
- **Fix**: Update `mmseg/models/utils/__init__.py` to include wavelet imports

**Issue 2**: SDM input dimensions not even
- **Fix**: Add padding in data pipeline or SDM forward:
  ```python
  if H % 2 != 0:
      x = F.pad(x, (0, 0, 0, 1))  # Pad height
  if W % 2 != 0:
      x = F.pad(x, (0, 1, 0, 0))  # Pad width
  ```

**Issue 3**: BoundarySemanticLoss returns NaN
- **Fix**: Check boundary prediction range with `torch.sigmoid(bd_pred).mean()`
- **Fix**: Ensure boundary GT has valid pixels (not all ignore_index)

**Issue 4**: AWM weights not updating
- **Fix**: Verify parameters are registered:
  ```python
  for name, param in model.named_parameters():
      if 'awm' in name:
          print(f'{name}: {param.requires_grad}')
  ```

---

## 📚 References

**Paper**: "Real-Time Long-Wave Infrared Semantic Segmentation With Adaptive Noise Reduction and Feature Fusion" (IEEE Access, 2025)

**Base Architecture**: PIDNet (https://github.com/XuJiacong/PIDNet)

**Framework**: MMSegmentation (https://github.com/open-mmlab/mmsegmentation)

---

## 📞 Support

For PROJECT-DETAIL REQUIRED items:
- NNS stripe noise parameters (λ range, frequency patterns)
- Boundary GT thickness and morphological operations
- Loss weight ratios from your experiments
- Any existing implementation fragments from previous project

Contact maintainer with specific questions.
