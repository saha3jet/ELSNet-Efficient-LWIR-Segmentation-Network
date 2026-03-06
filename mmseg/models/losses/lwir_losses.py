# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright (c) 2026 Haejun Bae.
# This file has been modified from the original MMSegmentation project.
# Licensed under the Apache License, Version 2.0.
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from .utils import weight_reduce_loss

from pytorch_wavelets import DWTForward


def _pack_wavelet(yl: Tensor, yh0: Tensor) -> Tensor:
    """Pack (yl, yh0) from pytorch_wavelets into a (B, 4C, H', W') tensor.

    yl:  (B, C,  H', W')  -> LL
    yh0: (B, C, 3, H', W') -> (LH, HL, HH) along dim=2
    """
    ll = yl
    lh = yh0[:, :, 0]
    hl = yh0[:, :, 1]
    hh = yh0[:, :, 2]
    return torch.cat([ll, lh, hl, hh], dim=1)


def _dwt_pack(
    x: Tensor,
    dwt: DWTForward,
    pad_mode: str = "replicate",
) -> Tensor:
    """Apply 1-level DWT (db1/reflect recommended) and pack into (B, 4C, H', W').

    Notes
    -----
    - For odd H/W, we pad bottom/right by 1px using pad_mode (default: replicate),
      consistent with SDM.forward_with_wave in elsnet.py.
    """
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=pad_mode)

    yl, yh = dwt(x)  # yh is a list (len=J=1)
    yh0 = yh[0]
    return _pack_wavelet(yl, yh0)


@MODELS.register_module()
class IMSELoss(nn.Module):
    """Inverse-MSE loss in wavelet domain (LiMSE).

    This implementation is aligned with SDM:
      - pytorch_wavelets DWTForward
      - wave=db1 (Haar), mode=reflect
      - pad_mode=replicate for odd H/W

    Signature kept compatible with existing ELSEncoderDecoder.loss usage.
    """

    def __init__(
        self,
        inverse: bool = True,
        per_channel: bool = False,
        eps: float = 1e-6,
        inverse_clip_max: Optional[float] = None,
        inverse_transform: str = "none",
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_imse",
        # wavelet settings (should match SDM)
        wave: str = "db1",
        mode: str = "reflect",
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.inverse = inverse
        self.per_channel = per_channel
        self.eps = eps
        self.inverse_clip_max = inverse_clip_max
        self.inverse_transform = inverse_transform
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name

        self.pad_mode = pad_mode
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)

    def forward(
        self,
        x_d: Tensor,
        x: Tensor,
        w_d: Optional[Tensor] = None,
        w_n: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        reduction = reduction_override if reduction_override else self.reduction

        if self.inverse:
            # w_d: wavelet-domain output of SDM(x)
            # w_n: wavelet-domain transform of NNS(x) or other negative sample
            if w_d is None or w_n is None:
                w_d = _dwt_pack(x_d, self.dwt, pad_mode=self.pad_mode)
                w_n = _dwt_pack(x, self.dwt, pad_mode=self.pad_mode)

            mse = F.mse_loss(w_d, w_n, reduction="none")
            if self.per_channel:
                # (B, C, H', W') -> (B, C)
                mse = mse.flatten(2).mean(dim=2)
            else:
                mse = mse.flatten(1).mean(dim=1, keepdim=True)

            inverse_mse = 1.0 / (mse + self.eps)

            if self.inverse_clip_max is not None:
                inverse_mse = torch.clamp(inverse_mse, max=self.inverse_clip_max)
            if self.inverse_transform == "log1p":
                inverse_mse = torch.log1p(inverse_mse)

            if weight is not None:
                weight = weight.float()
            loss = weight_reduce_loss(
                inverse_mse, weight=weight, reduction=reduction, avg_factor=avg_factor
            )
            return self.loss_weight * loss

        # Plain MSE (fallback; not used for LiMSE)
        loss = F.mse_loss(x_d, x, reduction="none")
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_


@MODELS.register_module()
class SoftContrastiveWaveletLoss(nn.Module):
    """Soft-contrastive loss in wavelet domain (Option 5-C).

    Anchor:   W_dn  = SDM_student( x_n )  (wavelet-packed)
    Positive: W_t   = SDM_teacher( x ) or stop-grad(SDM_student(x)) (wavelet-packed)
    Negative: W_n   = DWT( x_n )          (wavelet-packed, raw)

    Loss:
        d_pos = Charbonnier( HF(W_dn), HF(W_t) )
        d_neg = Charbonnier( HF(W_dn), HF(W_n) )
        L = softplus( d_pos - d_neg )

    Notes
    -----
    - Uses HF-only by default: (LH, HL, HH). LL is dropped.
    - Charbonnier is used as a robust distance (less sensitive than MSE).
    """

    def __init__(
        self,
        eps: float = 1e-3,
        use_hf_only: bool = True,
        reduction: str = "mean",
        rank_weight: float = 1.0, #
        loss_weight: float = 1.0,
        loss_name: str = "loss_scd",
    ):
        super().__init__()
        self.eps = float(eps)
        self.use_hf_only = bool(use_hf_only)
        self.reduction = reduction
        self.rank_weight = float(rank_weight) #
        self.loss_weight = float(loss_weight)
        self.loss_name_ = loss_name

    def _select_hf(self, w: Tensor) -> Tensor:
        if not self.use_hf_only:
            return w
        c4 = w.shape[1]
        assert c4 % 4 == 0, f"Expected packed wavelet with 4C channels, got {c4}"
        c = c4 // 4
        return w[:, c:, :, :]  # drop LL

    def _charbonnier_dist(self, a: Tensor, b: Tensor) -> Tensor:
        diff = a - b
        d = torch.sqrt(diff * diff + (self.eps * self.eps))
        # per-sample scalar distance
        return d.flatten(1).mean(dim=1, keepdim=True)

    def forward(
        self,
        # explicit tensors
        w_dn: Optional[Tensor] = None,
        w_t: Optional[Tensor] = None,
        w_n: Optional[Tensor] = None,
        # or dict carrier
        denoise_aux: Optional[Dict[str, Any]] = None,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        reduction = reduction_override if reduction_override else self.reduction

        if denoise_aux is not None:
            w_dn = denoise_aux.get("w_dn", w_dn)
            w_t = denoise_aux.get("w_t", w_t)
            w_n = denoise_aux.get("w_n", w_n)

        if w_dn is None or w_t is None or w_n is None:
            raise ValueError(
                "SoftContrastiveWaveletLoss requires (w_dn, w_t, w_n) or denoise_aux dict."
            )

        a = self._select_hf(w_dn)
        p = self._select_hf(w_t)
        n = self._select_hf(w_n)

        d_pos = self._charbonnier_dist(a, p)
        d_neg = self._charbonnier_dist(a, n)

        # loss_vec = F.softplus(d_pos - d_neg)
        loss_vec = d_pos + self.rank_weight * F.softplus(d_pos - d_neg)

        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss_vec, weight=weight, reduction=reduction, avg_factor=avg_factor)
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_


@MODELS.register_module()
class LowSemanticLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_ls",
        ignore_index: int = 255,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.ignore_index = ignore_index

    def forward(
        self,
        seg_logit: Tensor,
        seg_label: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        reduction = reduction_override if reduction_override else self.reduction
        loss = F.cross_entropy(
            seg_logit, seg_label, reduction="none", ignore_index=self.ignore_index
        )
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor
        )
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_


@MODELS.register_module()
class BoundarySemanticLoss(nn.Module):
    def __init__(
        self,
        threshold: float = 0.8,
        mode: str = "hard",
        soft_power: float = 1.0,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        loss_name: str = "loss_bas",
        ignore_index: int = 255,
    ):
        super().__init__()
        self.threshold = threshold
        self.mode = mode
        self.soft_power = soft_power
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name
        self.ignore_index = ignore_index

    def forward(
        self,
        seg_logit: Tensor,
        seg_label: Tensor,
        bd_pred: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[float] = None,
        reduction_override: Optional[str] = None,
        **kwargs,
    ) -> Tensor:
        reduction = reduction_override if reduction_override else self.reduction

        bd_prob = torch.sigmoid(bd_pred[:, 0, :, :])

        if self.mode == "soft":
            loss_map = F.cross_entropy(
                seg_logit, seg_label, reduction="none", ignore_index=self.ignore_index
            )
            valid = (seg_label != self.ignore_index).float()
            soft_weight = bd_prob.pow(self.soft_power) * valid
            denom = torch.clamp(soft_weight.sum(), min=self.eps)
            loss = (loss_map * soft_weight).sum() / denom
            return self.loss_weight * loss

        filler = seg_label.new_full(seg_label.shape, self.ignore_index)
        masked_label = torch.where(bd_prob > self.threshold, seg_label, filler)

        if not torch.any(masked_label != self.ignore_index):
            return seg_logit.sum() * 0.0

        loss = F.cross_entropy(
            seg_logit, masked_label, reduction="none", ignore_index=self.ignore_index
        )
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor
        )
        return self.loss_weight * loss

    @property
    def loss_name(self):
        return self.loss_name_
