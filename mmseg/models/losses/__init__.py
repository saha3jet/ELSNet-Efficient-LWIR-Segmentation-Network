# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright (c) 2026 Haejun Bae.
# This file has been modified from the original MMSegmentation project.
# Licensed under the Apache License, Version 2.0.
from .accuracy import Accuracy, accuracy
from .boundary_loss import BoundaryLoss
from .cross_entropy_loss import (
    CrossEntropyLoss,
    binary_cross_entropy,
    cross_entropy,
    mask_cross_entropy,
)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .huasdorff_distance_loss import HuasdorffDisstanceLoss
from .lovasz_loss import LovaszLoss
from .lwir_losses import BoundarySemanticLoss, IMSELoss, LowSemanticLoss, SoftContrastiveWaveletLoss
from .ohem_cross_entropy_loss import OhemCrossEntropy
from .silog_loss import SiLogLoss
from .tversky_loss import TverskyLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

__all__ = [
    "accuracy",
    "Accuracy",
    "cross_entropy",
    "binary_cross_entropy",
    "mask_cross_entropy",
    "CrossEntropyLoss",
    "reduce_loss",
    "weight_reduce_loss",
    "weighted_loss",
    "LovaszLoss",
    "DiceLoss",
    "FocalLoss",
    "TverskyLoss",
    "OhemCrossEntropy",
    "BoundaryLoss",
    "HuasdorffDisstanceLoss",
    "SiLogLoss",
    "IMSELoss",
    "LowSemanticLoss",
    "BoundarySemanticLoss",
    "SoftContrastiveWaveletLoss",
]
