# Copyright (c) OpenMMLab. All rights reserved.
# Modifications Copyright (c) 2026 Haejun Bae.
# This file has been modified from the original MMSegmentation project.
# Licensed under the Apache License, Version 2.0.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SODADataset(BaseSegDataset):
    """SODA dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for SODA dataset.
    """
    METAINFO = dict(
        classes=('background', 'person', 'building', 'tree', 'road', 'pole',
                 'grass', 'door', 'table', 'chair',
                 'car', 'bicycle', 'lamp', 'monitor', 'trafficCone', 'trash can', 'animal',
                 'fence', 'sky', 'river', 'sidewalk'),             
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
                 [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                 [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
                 [0, 64, 128], [0, 192, 0], [128, 192, 0], [0, 64, 128]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
