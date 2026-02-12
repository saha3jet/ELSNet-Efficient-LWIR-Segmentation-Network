# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def _haar_dwt(x: Tensor) -> Tensor:
    x_even_row = x[:, :, 0::2, :]
    x_odd_row = x[:, :, 1::2, :]

    ll = (
        x_even_row[:, :, :, 0::2]
        + x_odd_row[:, :, :, 0::2]
        + x_even_row[:, :, :, 1::2]
        + x_odd_row[:, :, :, 1::2]
    ) * 0.5
    lh = (
        x_even_row[:, :, :, 0::2]
        + x_odd_row[:, :, :, 0::2]
        - x_even_row[:, :, :, 1::2]
        - x_odd_row[:, :, :, 1::2]
    ) * 0.5
    hl = (
        x_even_row[:, :, :, 0::2]
        - x_odd_row[:, :, :, 0::2]
        + x_even_row[:, :, :, 1::2]
        - x_odd_row[:, :, :, 1::2]
    ) * 0.5
    hh = (
        x_even_row[:, :, :, 0::2]
        - x_odd_row[:, :, :, 0::2]
        - x_even_row[:, :, :, 1::2]
        + x_odd_row[:, :, :, 1::2]
    ) * 0.5

    return torch.cat([ll, lh, hl, hh], dim=1)


@MODELS.register_module()
class IMSELoss(nn.Module):
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
            if w_d is None or w_n is None:
                h, w = x_d.shape[-2:]
                pad_h = h % 2
                pad_w = w % 2
                if pad_h or pad_w:
                    x_d = F.pad(x_d, (0, pad_w, 0, pad_h), mode="replicate")
                    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
                w_d = _haar_dwt(x_d)
                w_n = _haar_dwt(x)

            mse = F.mse_loss(w_d, w_n, reduction="none")
            if self.per_channel:
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
                inverse_mse,
                weight=weight,
                reduction=reduction,
                avg_factor=avg_factor,
            )
            return self.loss_weight * loss

        loss = F.mse_loss(x_d, x, reduction="none")
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
