# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright (c) 2026 Haejun Bae.
# This file has been modified from the original MMSegmentation project.
# Licensed under the Apache License, Version 2.0.
import math
import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from pytorch_wavelets import DWTForward, DWTInverse
from ..utils import DAPPM, PAPPM, BasicBlock, Bottleneck
from .pidnet import PagFM, Bag, LightBag


class SDM(BaseModule):
    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 64,
        num_layers: int = 3,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        init_cfg: OptConfigType = None,
        # wavelet 설정
        wave: str = "db1",          # Haar == db1
        mode: str = "reflect",      # attention/경계 민감이면 reflect/symmetric 권장
        pad_mode: str = "replicate" # 기존 코드와 동일한 입력 pad 방식
    ):
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.pad_mode = pad_mode

        # 1-level DWT/IDWT
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

        # conv는 기존과 동일하게 "채널 concat된 4C" 입력을 처리
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

    @staticmethod
    def _pack_wavelet(yl: Tensor, yh0: Tensor) -> Tensor:
        """
        (yl, yh0) -> (B, 4C, H', W')
        yl:  (B, C,  H', W')  (LL)
        yh0: (B, C, 3, H', W') (LH, HL, HH)
        """
        # yh0[:, :, 0] = LH, 1 = HL, 2 = HH (일반적인 pytorch_wavelets 포맷)
        ll = yl
        lh = yh0[:, :, 0]
        hl = yh0[:, :, 1]
        hh = yh0[:, :, 2]
        return torch.cat([ll, lh, hl, hh], dim=1)

    def _unpack_wavelet(self, w: Tensor) -> Tuple[Tensor, Tensor]:
        """
        (B, 4C, H', W') -> (yl, yh0)
        """
        b, c4, h, w_ = w.shape
        assert c4 == 4 * self.in_channels, (
            f"Expected {4*self.in_channels} channels, got {c4}. "
            f"(Check SDM.in_channels vs input channel count.)"
        )
        c = self.in_channels

        ll = w[:, :c]
        lh = w[:, c:2*c]
        hl = w[:, 2*c:3*c]
        hh = w[:, 3*c:4*c]

        yl = ll
        yh0 = torch.stack([lh, hl, hh], dim=2)  # (B, C, 3, H', W')
        return yl, yh0

    @staticmethod
    def _match_hw(x: Tensor, target_h: int, target_w: int, pad_mode: str = "replicate") -> Tensor:
        """
        출력 크기가 입력 크기와 다르면 최소한으로 crop/pad 해서 맞춤.
        - 대부분의 mismatch는 1px 정도라 crop이 가장 덜 침습적임.
        """
        h, w = x.shape[-2], x.shape[-1]

        # crop (bottom/right)
        if h > target_h or w > target_w:
            x = x[..., :target_h, :target_w]
            h, w = x.shape[-2], x.shape[-1]

        # pad (bottom/right)
        if h < target_h or w < target_w:
            pad_bottom = target_h - h
            pad_right = target_w - w
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)

        return x


    def forward_wave(self, x: Tensor) -> Tensor:
        """Return wavelet-domain output W_d without reconstructing x_d.

        Useful for the *second* SDM pass in contrastive denoising (5-C),
        since it avoids IDWT reconstruction.
        """
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
        assert x.shape[1] == self.in_channels, (
            f"SDM configured with in_channels={self.in_channels}, but input has C={x.shape[1]}"
        )

        orig_h, orig_w = x.shape[-2], x.shape[-1]
        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)

        yl, yh = self.dwt(x)
        yh0 = yh[0]
        w_x = self._pack_wavelet(yl, yh0)
        w_d = self.conv_stack(w_x)
        return w_d

    def forward_with_wave(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        returns:
          x_d: 복원된 feature (B, C, H, W)
          w_d: wavelet domain feature (B, 4C, H', W')
        """
        assert x.dim() == 4, f"Expected 4D input (B,C,H,W), got {x.dim()}D"
        assert x.shape[1] == self.in_channels, (
            f"SDM configured with in_channels={self.in_channels}, "
            f"but input has C={x.shape[1]}"
        )

        orig_h, orig_w = x.shape[-2], x.shape[-1]

        # 기존 구현과 동일: 홀수면 replicate padding으로 짝수 맞춤
        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.pad_mode)

        # DWT
        yl, yh = self.dwt(x)   # yh: list length 1 (J=1)
        yh0 = yh[0]

        # (yl, yh0) -> (B, 4C, H', W') 로 pack 후 conv 처리
        w_x = self._pack_wavelet(yl, yh0)
        w_d = self.conv_stack(w_x)

        # conv 출력 다시 (yl, yh0)로 unpack
        yl_d, yh0_d = self._unpack_wavelet(w_d)

        # IDWT (원래 크기 복원)
        x_d = self.idwt((yl_d, [yh0_d]))

        # 입력에서 pad했다면 crop으로 원복
        if pad_h or pad_w:
            x_d = x_d[..., :orig_h, :orig_w]

        # 혹시 모드/구현 차이로 1px mismatch가 남는 경우 안전장치
        if x_d.shape[-2] != orig_h or x_d.shape[-1] != orig_w:
            x_d = self._match_hw(x_d, orig_h, orig_w, pad_mode=self.pad_mode)

        return x_d, w_d

    def forward(self, x: Tensor) -> Tensor:
        x_d, _ = self.forward_with_wave(x)
        return x_d


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


class AWM(nn.Module):
    """
    AWM with:
      - safer boundary mode for attention maps (default: reflect)
      - size safeguard: crop/pad ONLY if output spatial size mismatches input
    """
    def __init__(self, wave: str = "db1", mode: str = "reflect", pad_mode_if_needed: str = "replicate"):
        super().__init__()

        # Learnable scalar weights
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta  = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.omega = nn.Parameter(torch.tensor(0.5))

        # 1-level DWT/IDWT
        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

        # For the (rare) case output is smaller than input
        self.pad_mode_if_needed = pad_mode_if_needed

    @staticmethod
    def _match_hw(x: Tensor, target_h: int, target_w: int, pad_mode: str = "replicate") -> Tensor:
        """Match spatial size by bottom/right crop or bottom/right pad."""
        h, w = x.shape[-2], x.shape[-1]

        # If bigger: crop (bottom/right)
        if h > target_h or w > target_w:
            x = x[..., :target_h, :target_w]
            h, w = x.shape[-2], x.shape[-1]

        # If smaller: pad (bottom/right)
        if h < target_h or w < target_w:
            pad_bottom = target_h - h
            pad_right = target_w - w
            # pad = (left, right, top, bottom)
            x = F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode)

        return x

    def forward(self, fmap: Tensor) -> Tensor:
        """
        fmap: (B, C, H, W)
        returns: (B, C, H, W)  (enforced)
        """
        target_h, target_w = fmap.shape[-2], fmap.shape[-1]

        yl, yh = self.dwt(fmap)          # yl: LL, yh[0]: (B,C,3,H',W') with (LH,HL,HH)
        yh0 = yh[0]

        yl_w = self.alpha * yl

        yh0_w = yh0.clone()
        yh0_w[:, :, 0] = self.beta  * yh0[:, :, 0]   # LH
        yh0_w[:, :, 1] = self.gamma * yh0[:, :, 1]   # HL
        yh0_w[:, :, 2] = self.omega * yh0[:, :, 2]   # HH

        out = self.idwt((yl_w, [yh0_w]))  # nominally (B,C,H,W), but can be off by ~1px

        # Safety: enforce exact size only if mismatch
        if out.shape[-2] != target_h or out.shape[-1] != target_w:
            out = self._match_hw(out, target_h, target_w, pad_mode=self.pad_mode_if_needed)

        return out
   

class BEM(BaseModule):
    """
    Boundary Enhancement Module (BEM)
    - channel-wise max/avg pooling -> two spatial maps
    - AWM on each map
    - concat -> conv -> sigmoid -> multiply to f_b
    - residual add
    - ECA
    """
    def __init__(
        self,
        in_channels: int,
        norm_cfg: OptConfigType = dict(type="BN"),
        wave: str = "db1",
        mode: str = "reflect",
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)

        self.awm_max = AWM(wave=wave, mode=mode)
        self.awm_avg = AWM(wave=wave, mode=mode)

        self.spatial_conv = ConvModule(
            2, 1, kernel_size=7, padding=3, norm_cfg=norm_cfg, act_cfg=None
        )
        self.eca = ECALayer(in_channels)

    def forward(self, f_b: Tensor) -> Tensor:
        # f_b: (B, C, H, W)
        max_pool = torch.max(f_b, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(f_b, dim=1, keepdim=True)    # (B, 1, H, W)

        # AWM outputs are already (B, 1, H, W) due to IDWT reconstruction
        max_weighted = self.awm_max(max_pool)
        avg_weighted = self.awm_avg(avg_pool)

        spatial_input = torch.cat([max_weighted, avg_weighted], dim=1)  # (B, 2, H, W)
        spatial_attn = torch.sigmoid(self.spatial_conv(spatial_input))  # (B, 1, H, W)

        f_b_attn = f_b * spatial_attn
        f_b_res = f_b + f_b_attn
        return self.eca(f_b_res)


class MSFM(BaseModule):
    def __init__(
        self,
        hs_channels: int,
        ls_channels: int,
        b_channels: int,
        out_channels: int,
        ls_in_channels: int = None,
        norm_cfg: OptConfigType = dict(type="BN"),
        act_cfg: OptConfigType = dict(type="ReLU", inplace=True),
        align_corners: bool = False,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.align_corners = align_corners

        if ls_in_channels is None:
            ls_in_channels = ls_channels
        self.ls_align = (
            ConvModule(
                ls_in_channels,
                ls_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
            if ls_in_channels != ls_channels
            else nn.Identity()
        )

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
        self.pag = PagFM(
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
        # NEW: cheap channel alignment for f_ls (downsample -> 1x1 conv -> upsample)
        orig_size = f_ls.shape[2:]
        small_size = f_hs.shape[2:]
        f_ls_small = F.interpolate(
            f_ls,
            size=small_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        f_ls_small = self.ls_align(f_ls_small)
        f_ls = F.interpolate(
            f_ls_small,
            size=orig_size,
            mode="bilinear",
            align_corners=self.align_corners,
        )
        
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
    """Negative Noise Sample (NNS) generator (Option 2.1).

    Sensor-like fixed-pattern noise (FPN) approximation:
      - 1D correlated noise along rows/cols (irregular stripe-like pattern)
      - additive offset + multiplicative gain

    Notes
    -----
    - The legacy args (amplitude/freq_*) are kept for config compatibility,
      but the actual generation is NOT sinusoidal anymore.
    - For odd H/W, we do NOT need to pad here; SDM/loss handle padding for DWT.
    """

    def __init__(
        self,
        lambda_min: float = 0.05,
        lambda_max: float = 0.20,
        # legacy args (kept for config compatibility)
        amplitude: float = 0.15,
        freq_min: float = 2.0,   # legacy (unused)
        freq_max: float = 12.0,  # legacy (unused)
        direction: str = "both",
        # new args
        offset_std: Optional[float] = None,
        gain_std: Optional[float] = None,
        corr_kernel_size: int = 21,
        corr_sigma: Optional[float] = None,
        eps: float = 1e-6,
        init_cfg: OptConfigType = None,
    ):
        super().__init__(init_cfg)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.amplitude = float(amplitude)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.direction = direction

        # If not explicitly provided, map legacy amplitude to std scales
        self.offset_std = float(offset_std) if offset_std is not None else float(amplitude)
        self.gain_std = float(gain_std) if gain_std is not None else float(amplitude)

        self.corr_kernel_size = int(corr_kernel_size)
        # default sigma: ~k/6 is a common heuristic
        if corr_sigma is None:
            self.corr_sigma = max(self.corr_kernel_size / 6.0, 1.0)
        else:
            self.corr_sigma = float(corr_sigma)

        self.eps = float(eps)

    def _gaussian_kernel_1d(self, device, dtype) -> Tensor:
        k = self.corr_kernel_size
        sigma = self.corr_sigma
        x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / (kernel.sum() + self.eps)
        return kernel.view(1, 1, k)  # (out_ch=1, in_ch=1, k)

    def _correlated_1d(self, b: int, c: int, length: int, device, dtype) -> Tensor:
        """Return (B, C, 1, L) correlated noise with ~N(0,1) per sample/channel."""
        # white noise
        n = torch.randn((b * c, 1, length), device=device, dtype=dtype)

        # low-pass correlation
        kernel = self._gaussian_kernel_1d(device=device, dtype=dtype)
        n = F.conv1d(n, kernel, padding=self.corr_kernel_size // 2)

        # normalize per (B,C)
        n = n.view(b, c, 1, length)
        n = n - n.mean(dim=-1, keepdim=True)
        n = n / (n.std(dim=-1, keepdim=True) + self.eps)
        return n

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        device, dtype = x.device, x.dtype

        lam = torch.empty((b, 1, 1, 1), device=device, dtype=dtype).uniform_(
            self.lambda_min, self.lambda_max
        )

        # accumulate offset/gain contributions (row/col)
        offset = torch.zeros((b, c, h, w), device=device, dtype=dtype)
        gain = torch.zeros((b, c, h, w), device=device, dtype=dtype)

        if self.direction in ("vertical", "both"):
            # column-wise patterns: (B,C,1,W) -> broadcast to H
            col_off = self._correlated_1d(b, c, w, device, dtype) * self.offset_std
            col_gain = self._correlated_1d(b, c, w, device, dtype) * self.gain_std
            offset = offset + col_off.expand(b, c, h, w)
            gain = gain + col_gain.expand(b, c, h, w)

        if self.direction in ("horizontal", "both"):
            # row-wise patterns: (B,C,1,H) -> (B,C,H,1) -> broadcast to W
            row_off = self._correlated_1d(b, c, h, device, dtype) * self.offset_std
            row_gain = self._correlated_1d(b, c, h, device, dtype) * self.gain_std
            row_off = row_off.transpose(-1, -2)   # (B,C,H,1)
            row_gain = row_gain.transpose(-1, -2) # (B,C,H,1)
            offset = offset + row_off.expand(b, c, h, w)
            gain = gain + row_gain.expand(b, c, h, w)

        # sensor-like injection: gain + offset
        x_n = x * (1.0 + lam * gain) + lam * offset

        # preserve dynamic range
        x_min = torch.amin(x, dim=(2, 3), keepdim=True).detach()
        x_max = torch.amax(x, dim=(2, 3), keepdim=True).detach()
        return torch.clamp(x_n, x_min, x_max)


@MODELS.register_module()
class ELSNet(BaseModule):
    def __init__(self,
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
                # SDM teacher (EMA) for denoising contrastive learning
                use_sdm_teacher: bool = False,
                sdm_teacher_momentum: float = 0.999,
                init_cfg: OptConfigType = None,
                **kwargs):
        super().__init__(init_cfg)
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners

        # stem layer
        self.stem = self._make_stem_layer(in_channels, channels, num_stem_blocks)
        self.relu = nn.ReLU()

        # I Branch
        self.i_branch_layers = nn.ModuleList()
        for i in range(3):
            self.i_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2 ** (i + 1),
                    channels=channels * 8 if i > 0 else channels * 4,
                    num_blocks=num_branch_blocks if i < 2 else 2,
                    stride=2,
                )
            )

        # P Branch
        self.p_branch_layers = nn.ModuleList()
        for i in range(3):
            self.p_branch_layers.append(
                self._make_layer(
                    block=BasicBlock if i < 2 else Bottleneck,
                    in_channels=channels * 2,
                    channels=channels * 2,
                    num_blocks=num_stem_blocks if i < 2 else 1))
        self.compression_1 = ConvModule(
            channels * 4,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.compression_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None)
        self.pag_1 = PagFM(channels * 2, channels)
        self.pag_2 = PagFM(channels * 2, channels)

        if num_stem_blocks == 2:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels),
                self._make_layer(Bottleneck, channels, channels, 1),
            ])
            channel_expand = 1
            spp_module = PAPPM
            dfm_module = LightBag
            act_cfg_dfm = None
        else:
            self.d_branch_layers = nn.ModuleList([
                self._make_single_layer(BasicBlock, channels * 2, channels * 2),
                self._make_single_layer(BasicBlock, channels * 2, channels * 2),
            ])
            channel_expand = 2
            spp_module = DAPPM
            dfm_module = Bag
            act_cfg_dfm = act_cfg

        self.diff_1 = ConvModule(
            channels * 4,
            channels * channel_expand,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.diff_2 = ConvModule(
            channels * 8,
            channels * 2,
            kernel_size=3,
            padding=1,
            bias=False,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.spp = spp_module(channels * 16, ppm_channels, channels * 4, num_scales=5)
        self.dfm = dfm_module(
            channels * 4, channels * 4, norm_cfg=norm_cfg, act_cfg=act_cfg_dfm
        )

        self.d_branch_layers.append(
            self._make_layer(Bottleneck, channels * 2, channels * 2, 1)
        )

        # ----- (B) ELSNet 고유 모듈 추가(기존 코드 그대로) -----
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

        # ---- SDM teacher (EMA) ----
        self.use_sdm_teacher = bool(use_sdm_teacher)
        self.sdm_teacher_momentum = float(sdm_teacher_momentum)
        if self.use_sdm_teacher:
            self.sdm_teacher = copy.deepcopy(self.sdm)
            self.sdm_teacher.eval()
            for p in self.sdm_teacher.parameters():
                p.requires_grad_(False)
        else:
            self.sdm_teacher = None


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
                ls_in_channels=channels * 4,
                **msfm_base_args,
            )

        nns_args = dict()
        if nns_cfg is not None:
            nns_args.update(nns_cfg)
        self.nns = NNSGenerator(**nns_args)

    def _make_stem_layer(self, in_channels: int, channels: int, num_blocks: int) -> nn.Sequential:
        layers = [
            ConvModule(
                in_channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            ConvModule(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        ]

        layers.append(self._make_layer(BasicBlock, channels, channels, num_blocks))
        layers.append(nn.ReLU())
        layers.append(self._make_layer(BasicBlock, channels, channels * 2, num_blocks, stride=2))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _make_layer(
        self,
        block,
        in_channels: int,
        channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
            )

        layers = [block(in_channels, channels, stride, downsample)]
        in_channels = channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels,
                    channels,
                    stride=1,
                    act_cfg_out=None if i == num_blocks - 1 else self.act_cfg,
                )
            )
        return nn.Sequential(*layers)

    def _make_single_layer(
        self,
        block,
        in_channels: int,
        channels: int,
        stride: int = 1,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or in_channels != channels * block.expansion:
            downsample = ConvModule(
                in_channels,
                channels * block.expansion,
                kernel_size=1,
                stride=stride,
                norm_cfg=self.norm_cfg,
                act_cfg=None,
            )
        return block(in_channels, channels, stride, downsample, act_cfg_out=None)

    # ---------- init_weights: shape 일치만 로드(첫 conv mismatch 자동 스킵) ----------
    def init_weights(self):
        # 기본 kaiming init (PIDNet 방식 유지)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.init_cfg is None:
            return

        assert isinstance(self.init_cfg, dict) and "checkpoint" in self.init_cfg, (
            "ELSNet.init_cfg must be a dict and include `checkpoint`."
        )

        ckpt = CheckpointLoader.load_checkpoint(self.init_cfg["checkpoint"], map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)

        model_sd = self.state_dict()
        load_sd = {k: v for k, v in sd.items()
                if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
        self.load_state_dict(load_sd, strict=False)

    # ---------- 기존 ELSNet 기능들 ----------
    def generate_nns(self, x: Tensor) -> Tensor:
        return self.nns(x)

    def raw_wavelet_pack(self, x: Tensor) -> Tensor:
        """Compute packed 1-level wavelet features W(x) using SDM's DWT settings.

        Returns:
            w: (B, 4C, H', W') where channels are [LL, LH, HL, HH].
        """
        assert hasattr(self, "sdm") and hasattr(self.sdm, "dwt"), "SDM must provide DWTForward as `sdm.dwt`."
        orig_h, orig_w = x.shape[-2], x.shape[-1]
        pad_h = orig_h % 2
        pad_w = orig_w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.sdm.pad_mode)

        yl, yh = self.sdm.dwt(x)
        yh0 = yh[0]
        return SDM._pack_wavelet(yl, yh0)

    @torch.no_grad()
    def update_sdm_teacher(self, momentum=None, update_buffers=True):
        """EMA update for SDM teacher.

        Call this AFTER optimizer.step() in training loop / hook.
        """
        if self.sdm_teacher is None:
            return

        m = self.sdm_teacher_momentum if momentum is None else float(momentum)

        # parameters
        for (n_t, p_t), (n_s, p_s) in zip(
            self.sdm_teacher.named_parameters(), self.sdm.named_parameters()
        ):
            if p_t.data.shape != p_s.data.shape:
                continue
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

        # buffers는 옵션에 따라 실행
        if not update_buffers:
            return

        for (n_t, b_t), (n_s, b_s) in zip(
            self.sdm_teacher.named_buffers(), self.sdm.named_buffers()
        ):
            if b_t.data.shape != b_s.data.shape:
                continue
            if not torch.is_floating_point(b_t):
                b_t.data.copy_(b_s.data)
            else:
                b_t.data.mul_(m).add_(b_s.data, alpha=1.0 - m)

    @torch.no_grad()
    def teacher_denoise_with_wave(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward SDM teacher (if enabled), else return stop-grad student output."""
        if self.sdm_teacher is None:
            x_d, w_d = self.sdm.forward_with_wave(x)
            return x_d.detach(), w_d.detach()
        return self.sdm_teacher.forward_with_wave(x)


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
        x_d += F.interpolate(diff_i, size=[h_out, w_out], mode="bilinear",
                             align_corners=self.align_corners)
        if self.training:
            temp_p = x_p.clone()

        x_i = self.relu(self.i_branch_layers[1](x_i))
        x_p = self.p_branch_layers[1](self.relu(x_p))
        x_d = self.bem_2(self.d_branch_layers[1](self.relu(x_d)))

        comp_i = self.compression_2(x_i)
        x_p = self.pag_2(x_p, comp_i)
        diff_i = self.diff_2(x_i)
        x_d += F.interpolate(diff_i, size=[h_out, w_out], mode="bilinear",
                             align_corners=self.align_corners)

        if self.use_msfm:
            msfm_feat = self.msfm_1(x_i, x_p, x_d)
            msfm_feat = F.interpolate(msfm_feat, size=x_i.shape[2:], mode="bilinear",
                                      align_corners=self.align_corners)
            x_i = x_i + msfm_feat

        if self.training:
            temp_d = x_d.clone()

        x_i = self.i_branch_layers[2](x_i)
        x_p = self.p_branch_layers[2](self.relu(x_p))
        x_d = self.bem_3(self.d_branch_layers[2](self.relu(x_d)))

        x_i = self.spp(x_i)
        x_i = F.interpolate(x_i, size=[h_out, w_out], mode="bilinear",
                            align_corners=self.align_corners)

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
