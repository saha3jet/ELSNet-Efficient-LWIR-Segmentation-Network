# ELSNet: Efficient LWIR Segmentation Network

This is the official repository for the paper:

**[Real-Time Long-Wave Infrared Semantic Segmentation With Adaptive Noise Reduction and Feature Fusion](https://doi.org/10.1109/ACCESS.2025.3552782)**  
(IEEE Access, 2025)

The framework proposed in this work is named **ELSNet**.

## Overview

ELSNet is designed for real-time LWIR semantic segmentation.

Its core architecture includes:

- SDM for adaptive denoising
- BEM for boundary-aware enhancement
- MSFM for multi-stream feature fusion

These components are integrated in a real-time PID-style segmentation framework.

## Implementation Note

This repository implements the paper structure and additionally refines the NNS-related training path with an optional teacher-assisted mode.

---

Built on top of MMSegmentation.
