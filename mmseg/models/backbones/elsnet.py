# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from .pidnet import PIDNet


def haar_dwt(x: Tensor) -> Tensor:
    assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
    _, _, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0, (
        f"Input height and width must be even, got H={h}, W={w}"
    )

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


def haar_idwt(coeffs: Tensor) -> Tensor:
    assert coeffs.dim() == 4, f"Expected 4D input (B,C*4,H,W), got {coeffs.dim()}D"
    b, c4, h, w = coeffs.shape
    assert c4 % 4 == 0, f"Channels must be divisible by 4, got {c4}"
    c = c4 // 4

    ll = coeffs[:, :c]
    lh = coeffs[:, c : 2 * c]
    hl = coeffs[:, 2 * c : 3 * c]
    hh = coeffs[:, 3 * c :]

    x00 = ll + lh + hl + hh
    x01 = ll - lh + hl - hh
    x10 = ll + lh - hl - hh
    x11 = ll - lh - hl + hh

    top = torch.stack([x00, x01], dim=-1)
    bottom = torch.stack([x10, x11], dim=-1)
    out = torch.stack([top, bottom], dim=-2)
    out = out.reshape(b, c, h * 2, w * 2)
    return out * 0.5


class ECALayer(BaseModule):
    def __init__(
        self, channels: int, gamma: int = 2, b: int = 1, init_cfg: OptConfigType = None
    ):
        super().__init__(init_cfg)
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        return x * y.expand_as(x)


class AWM(BaseModule):
    def __init__(self, init_cfg: OptConfigType = None):
        super().__init__(init_cfg)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.omega = nn.Parameter(torch.tensor(0.5))

    def forward(self, feature_2d: Tensor) -> Tensor:
        coeffs = haar_dwt(feature_2d)
        _, c4, _, _ = coeffs.shape
        c = c4 // 4
        ll = coeffs[:, :c]
        lh = coeffs[:, c : 2 * c]
        hl = coeffs[:, 2 * c : 3 * c]
        hh = coeffs[:, 3 * c :]
        weighted = (
            self.alpha * ll + self.beta * lh + self.gamma * hl + self.omega * hh
        ) / 4.0
        return weighted


class SDM(BaseModule):
    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 64,
        num_layers: int = 3,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        wavelet_channels = in_channels * 4
        layers = []
        in_ch = wavelet_channels
        for i in range(num_layers):
            out_ch = mid_channels if i < num_layers - 1 else wavelet_channels
            use_act = act_cfg if i < num_layers - 1 else None
            layers.append(
                ConvModule(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    act_cfg=use_act,
                )
            )
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

    def forward_with_wave(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        w_x = haar_dwt(x)
        w_d = self.conv_stack(w_x)
        x_d = haar_idwt(w_d)

        if pad_h or pad_w:
            x_d = x_d[..., :h, :w]

        return x_d, w_d

    def forward(self, x: Tensor) -> Tensor:
        x_d, _ = self.forward_with_wave(x)

        return x_d


class BEM(BaseModule):
    def __init__(
        self,
        in_channels: int,
        norm_cfg: OptConfigType = dict(type="BN"),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.awm_max = AWM()
        self.awm_avg = AWM()
        self.spatial_conv = ConvModule(
            2, 1, kernel_size=7, padding=3, norm_cfg=norm_cfg, act_cfg=None
        )
        self.eca = ECALayer(in_channels)

    def forward(self, f_b: Tensor) -> Tensor:
        max_pool = torch.max(f_b, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(f_b, dim=1, keepdim=True)

        max_weighted = self.awm_max(max_pool)
        avg_weighted = self.awm_avg(avg_pool)

        max_up = F.interpolate(
            max_weighted, size=f_b.shape[2:], mode="bilinear", align_corners=False
        )
        avg_up = F.interpolate(
            avg_weighted, size=f_b.shape[2:], mode="bilinear", align_corners=False
        )

        spatial_input = torch.cat([max_up, avg_up], dim=1)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_input))
        f_b_attn = f_b * spatial_attn
        f_b_res = f_b + f_b_attn
        return self.eca(f_b_res)


class _PagFM(BaseModule):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        upsample_mode: str = "bilinear",
        norm_cfg: OptConfigType = dict(type="BN"),
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.upsample_mode = upsample_mode
        self.f_i = ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)
        self.f_p = ConvModule(in_channels, channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x_p: Tensor, x_i: Tensor) -> Tensor:
        f_i = self.f_i(x_i)
        f_i = F.interpolate(
            f_i, size=x_p.shape[2:], mode=self.upsample_mode, align_corners=False
        )
        f_p = self.f_p(x_p)
        sigma = torch.sigmoid(torch.sum(f_p * f_i, dim=1).unsqueeze(1))
        x_i = F.interpolate(
            x_i, size=x_p.shape[2:], mode=self.upsample_mode, align_corners=False
        )
        return sigma * x_i + (1 - sigma) * x_p


class MSFM(BaseModule):
    def __init__(
        self,
        hs_channels: int,
        ls_channels: int,
        b_channels: int,
        out_channels: int,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        align_corners: bool = False,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        self.v_conv = ConvModule(
            ls_channels + b_channels, ls_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )
        self.h_conv = ConvModule(
            ls_channels + b_channels, ls_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
        )

        self.downsample = ConvModule(
            ls_channels * 2,
            hs_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.hs_to_ls = ConvModule(
            hs_channels, ls_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None
        )
        self.hs_to_b = ConvModule(
            hs_channels, b_channels, kernel_size=1, norm_cfg=norm_cfg, act_cfg=None
        )
        self.pag = _PagFM(
            in_channels=ls_channels,
            channels=max(ls_channels // 2, 1),
            upsample_mode="bilinear",
            norm_cfg=norm_cfg,
        )

        self.fusion_conv = ConvModule(
            hs_channels + ls_channels + b_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, f_hs: Tensor, f_ls: Tensor, f_b: Tensor) -> Tensor:
        f_comb = torch.cat([f_ls, f_b], dim=1)

        v_pool = torch.mean(f_comb, dim=3, keepdim=True).expand_as(f_comb)
        f_v_attn = torch.sigmoid(self.v_conv(v_pool))

        h_pool = torch.mean(f_comb, dim=2, keepdim=True).expand_as(f_comb)
        f_h_attn = torch.sigmoid(self.h_conv(h_pool))

        f_v = f_ls * f_v_attn[:, : f_ls.shape[1]]
        f_h = f_ls * f_h_attn[:, : f_ls.shape[1]]
        f_els = torch.cat([f_v, f_h], dim=1)

        gate = torch.sigmoid(self.downsample(f_els))
        gate = F.interpolate(
            gate, size=f_hs.shape[2:], mode="bilinear", align_corners=self.align_corners
        )
        f_hs_gated = f_hs * gate

        f_hs_up = F.interpolate(
            f_hs_gated,
            size=f_ls.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        f_hs_up_ls = self.hs_to_ls(f_hs_up)
        f_global = self.pag(f_ls, f_hs_up_ls)

        f_hs_up_b = self.hs_to_b(f_hs_up)
        f_b_enhanced = f_b + f_hs_up_b

        fused = torch.cat([f_global, f_hs_up, f_b_enhanced], dim=1)
        return self.fusion_conv(fused)


class NNSGenerator(BaseModule):
    def __init__(
        self,
        lambda_min: float = 0.05,
        lambda_max: float = 0.20,
        amplitude: float = 0.15,
        freq_min: float = 2.0,
        freq_max: float = 12.0,
        direction: str = "both",
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.amplitude = amplitude
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.direction = direction

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        device = x.device
        dtype = x.dtype

        y = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1)
        z = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w)

        stripe = torch.zeros((b, c, h, w), device=device, dtype=dtype)

        if self.direction in ("horizontal", "both"):
            freq_h = torch.empty((b, c, 1, 1), device=device, dtype=dtype).uniform_(
                self.freq_min, self.freq_max
            )
            phase_h = torch.empty((b, c, 1, 1), device=device, dtype=dtype).uniform_(
                0.0, 2.0 * math.pi
            )
            stripe = stripe + torch.sin(2.0 * math.pi * freq_h * y + phase_h)

        if self.direction in ("vertical", "both"):
            freq_w = torch.empty((b, c, 1, 1), device=device, dtype=dtype).uniform_(
                self.freq_min, self.freq_max
            )
            phase_w = torch.empty((b, c, 1, 1), device=device, dtype=dtype).uniform_(
                0.0, 2.0 * math.pi
            )
            stripe = stripe + torch.sin(2.0 * math.pi * freq_w * z + phase_w)

        stripe = stripe * self.amplitude
        lam = torch.empty((b, 1, 1, 1), device=device, dtype=dtype).uniform_(
            self.lambda_min, self.lambda_max
        )

        x_n = x + lam * stripe
        x_min = torch.amin(x, dim=(2, 3), keepdim=True).detach()
        x_max = torch.amax(x, dim=(2, 3), keepdim=True).detach()
        return torch.clamp(x_n, x_min, x_max)


@MODELS.register_module()
class ELSNet(PIDNet):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 64,
        ppm_channels: int = 96,
        num_stem_blocks: int = 2,
        num_branch_blocks: int = 3,
        align_corners: bool = False,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        sdm_cfg: OptConfigType = None,
        bem_cfg: OptConfigType = None,
        use_msfm: bool = True,
        msfm_cfg: OptConfigType = None,
        nns_cfg: OptConfigType = None,
        init_cfg: OptConfigType = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            ppm_channels=ppm_channels,
            num_stem_blocks=num_stem_blocks,
            num_branch_blocks=num_branch_blocks,
            align_corners=align_corners,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
            **kwargs,
        )

        sdm_args = dict(
            in_channels=in_channels,
            mid_channels=64,
            num_layers=3,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        if sdm_cfg is not None:
            sdm_args.update(sdm_cfg)
        self.sdm = SDM(**sdm_args)

        d_stage1_channels = channels if num_stem_blocks == 2 else channels * 2
        d_stage2_channels = channels * 2
        d_stage3_channels = channels * 4

        bem_base_args = dict(norm_cfg=norm_cfg)
        if bem_cfg is not None:
            bem_base_args.update(bem_cfg)
        self.bem_1 = BEM(in_channels=d_stage1_channels, **bem_base_args)
        self.bem_2 = BEM(in_channels=d_stage2_channels, **bem_base_args)
        self.bem_3 = BEM(in_channels=d_stage3_channels, **bem_base_args)

        self.use_msfm = use_msfm
        if self.use_msfm:
            msfm_base_args = dict(
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                align_corners=align_corners,
            )
            if msfm_cfg is not None:
                msfm_base_args.update(msfm_cfg)
            self.msfm_1 = MSFM(
                hs_channels=channels * 8,
                ls_channels=channels * 2,
                b_channels=d_stage2_channels,
                out_channels=channels * 8,
                **msfm_base_args,
            )
            self.msfm_2 = MSFM(
                hs_channels=channels * 4,
                ls_channels=channels * 2,
                b_channels=d_stage3_channels,
                out_channels=channels * 4,
                **msfm_base_args,
            )

        nns_args = dict()
        if nns_cfg is not None:
            nns_args.update(nns_cfg)
        self.nns = NNSGenerator(**nns_args)

    def generate_nns(self, x: Tensor) -> Tensor:
        return self.nns(x)

    def forward_from_denoised(self, x_denoised: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        w_out = x_denoised.shape[-1] // 8
        h_out = x_denoised.shape[-2] // 8

        x = self.stem(x_denoised)

        x_i = self.relu(self.i_branch_layers[0](x))
        x_p = self.p_branch_layers[0](x)
        x_d = self.bem_1(self.d_branch_layers[0](x))
        temp_p = x_p
        temp_d = x_d

        comp_i = self.compression_1(x_i)
        x_p = self.pag_1(x_p, comp_i)
        diff_i = self.diff_1(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        if self.training:
            temp_p = x_p.clone()

        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.bem_2(self.d_branch_layers[1](self.relu(x_d)))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(
            diff_i,
            size=[h_out, w_out],
            mode="bilinear",
            align_corners=self.align_corners,
        )

        if self.use_msfm:
            msfm_feat = self.msfm_1(x_i, x_p, x_d)
            msfm_feat = F.interpolate(
                msfm_feat,
                size=x_i.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            x_i = x_i + msfm_feat

        if self.training:
            temp_d = x_d.clone()

        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.bem_3(self.d_branch_layers[2](self.relu(x_d)))

        x_i = self.spp(x_i)
        x_i = F.interpolate(
            x_i, size=[h_out, w_out], mode="bilinear", align_corners=self.align_corners
        )

        if self.use_msfm:
            x_i = x_i + self.msfm_2(x_i, x_p, x_d)

        out = self.dfm(x_p, x_i, x_d)
        if self.training:
            return temp_p, out, temp_d
        return out

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        x_d = self.sdm(x)
        return self.forward_from_denoised(x_d)

    def denoise_with_wave(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.sdm.forward_with_wave(x)
