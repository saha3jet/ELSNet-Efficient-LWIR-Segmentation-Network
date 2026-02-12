# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList
from .encoder_decoder import EncoderDecoder


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

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = dict()

        if (
            self.loss_imse is not None
            and hasattr(self.backbone, "sdm")
            and hasattr(self.backbone, "forward_from_denoised")
        ):
            if hasattr(self.backbone, "denoise_with_wave"):
                x_d, w_d = self.backbone.denoise_with_wave(inputs)
            else:
                x_d = self.backbone.sdm(inputs)
                w_d = None
            if hasattr(self.backbone, "generate_nns"):
                x_n = self.backbone.generate_nns(inputs)
            else:
                x_n = inputs

            x = self.backbone.forward_from_denoised(x_d)
            if self.with_neck:
                x = self.neck(x)

            losses["loss_imse"] = self.loss_imse(x_d, x_n, w_d=w_d)
        else:
            x = self.extract_feat(inputs)

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses
