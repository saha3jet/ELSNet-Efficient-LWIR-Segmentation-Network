# Copyright (c) OpenMMLab. All rights reserved.
"""ELSEncoderDecoder with Soft-Contrastive Denoising Loss (Option 5-C).

This file is a drop-in replacement for your current ELSEncoderDecoder.
It keeps backward compatibility with the original IMSELoss path.

Key changes:
- If `loss_imse` is `SoftContrastiveWaveletLoss`, it computes:
    * x_d, w_d = SDM(x)
    * x_n = NNS(x)
    * w_dn = SDM(x_n)      (wavelet-domain; gradients enabled)
    * w_n  = DWT(x_n)      (raw wavelet; no-grad)
    * w_t  = teacher(x) or stop-grad(w_d)
  and calls loss as: loss_imse(w_dn=w_dn, w_t=w_t, w_n=w_n)

- Otherwise it falls back to the original IMSELoss signature:
  loss_imse(x_d, x_n, w_d=w_d)

Notes:
- For teacher EMA update, use an MMEngine hook (separate file) or
  update teacher in your training loop.
"""

from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from .encoder_decoder import EncoderDecoder


def _raw_wavelet_pack_like_sdm(x: Tensor, sdm) -> Tensor:
    """Compute packed wavelet W = [LL, LH, HL, HH] using SDM's DWT settings.

    Assumes SDM exposes:
      - sdm.dwt : pytorch_wavelets.DWTForward(J=1)
      - sdm.pad_mode : padding mode for odd H/W (replicate)
      - sdm._pack_wavelet(yl, yh0)
    """
    h, w = x.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=getattr(sdm, "pad_mode", "replicate"))
    yl, yh = sdm.dwt(x)
    yh0 = yh[0]
    return sdm._pack_wavelet(yl, yh0)


@MODELS.register_module()
class ELSEncoderDecoder(EncoderDecoder):
    def __init__(
        self,
        backbone: ConfigType,
        decode_head: ConfigType,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        loss_imse: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.loss_imse = MODELS.build(loss_imse) if loss_imse is not None else None

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def _compute_denoise_loss(self, inputs: Tensor) -> (Tensor, dict):
        """Compute denoise preprocessing + denoise loss.

        Returns
        -------
        x_feat : backbone features computed from denoised input (for decode head)
        denoise_losses : dict with denoise loss terms
        """
        denoise_losses = {}

        # (1) SDM pass #1: x -> x_d, w_d
        if hasattr(self.backbone, "denoise_with_wave"):
            x_d, w_d = self.backbone.denoise_with_wave(inputs)
        else:
            # fallback
            x_d = self.backbone.sdm(inputs)
            w_d = None

        # (2) NNS generation: x -> x_n
        if hasattr(self.backbone, "generate_nns"):
            x_n = self.backbone.generate_nns(inputs)
        else:
            x_n = inputs

        # (3) Choose denoise loss type
        loss_type = self.loss_imse.__class__.__name__
        is_soft_contrastive = (loss_type == "SoftContrastiveWaveletLoss")

        if is_soft_contrastive:
            # Anchor: W_dn = SDM(x_n) (wavelet-domain), grad enabled
            if hasattr(self.backbone, "sdm") and hasattr(self.backbone.sdm, "forward_wave"):
                w_dn = self.backbone.sdm.forward_wave(x_n)
            else:
                # less efficient fallback
                _, w_dn = self.backbone.sdm.forward_with_wave(x_n)

            with torch.no_grad():
                # Positive: teacher(x) or stop-grad student(x)
                if getattr(self.backbone, "sdm_teacher", None) is not None:
                    # teacher exists
                    _, w_t = self.backbone.sdm_teacher.forward_with_wave(inputs)
                else:
                    # no teacher -> stop-grad of student w_d
                    if w_d is None:
                        _, w_d_tmp = self.backbone.sdm.forward_with_wave(inputs)
                        w_t = w_d_tmp.detach()
                    else:
                        w_t = w_d.detach()

                # Negative: raw wavelet transform of x_n using SDM DWT
                if hasattr(self.backbone, "raw_wavelet_pack"):
                    w_n = self.backbone.raw_wavelet_pack(x_n)
                else:
                    w_n = _raw_wavelet_pack_like_sdm(x_n, self.backbone.sdm)

            denoise_losses["loss_imse"] = self.loss_imse(w_dn=w_dn, w_t=w_t, w_n=w_n)

        else:
            # Original path (IMSELoss): compare SDM(x) vs NNS(x)
            denoise_losses["loss_imse"] = self.loss_imse(x_d, x_n, w_d=w_d)

        # (4) Backbone feature extraction from denoised input
        x_feat = self.backbone.forward_from_denoised(x_d)
        if self.with_neck:
            x_feat = self.neck(x_feat)

        return x_feat, denoise_losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        if (
            self.loss_imse is not None
            and hasattr(self.backbone, "sdm")
            and hasattr(self.backbone, "forward_from_denoised")
        ):
            x, denoise_losses = self._compute_denoise_loss(inputs)
            losses.update(denoise_losses)
        else:
            x = self.extract_feat(inputs)

        # decode head
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        # auxiliary head
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
