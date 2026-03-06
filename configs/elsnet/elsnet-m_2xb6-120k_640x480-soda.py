# Modifications Copyright (c) 2026 Haejun Bae.
# Derived from MMSegmentation configuration and licensed under Apache License 2.0.
_base_ = [
    "../_base_/datasets/soda_640x480.py",
    "../_base_/default_runtime.py",
]
load_from = None
crop_size = (480, 640)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675],
    std=[58.395],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
)

norm_cfg = dict(type="SyncBN", requires_grad=True)

model = dict(
    type="ELSEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="ELSNet",
        in_channels=1,
        channels=64,
        ppm_channels=96,
        num_stem_blocks=2,
        num_branch_blocks=3,
        align_corners=False,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU", inplace=True),
        use_msfm=True,
        sdm_cfg=dict(in_channels=1, mid_channels=64, num_layers=3),
        nns_cfg=dict(
            lambda_min=0.05,
            lambda_max=0.20,
            amplitude=0.15,
            freq_min=2.0,
            freq_max=12.0,
            direction="both",),
        use_sdm_teacher=True,
        sdm_teacher_momentum=0.999,
    ),
    decode_head=dict(
        type="ELSHead",
        in_channels=256,
        channels=128,
        num_classes=21,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU", inplace=True),
        align_corners=True,
        bas_threshold=0.8,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                class_weight=None,
                loss_weight=0.4,
            ),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=None,
                loss_weight=1.0,
            ),
            dict(type="BoundaryLoss", loss_weight=20.0),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=None,
                loss_weight=1.0,
            ),
        ],
    ),
    loss_imse=dict(
    type="SoftContrastiveWaveletLoss",
    eps=1e-3,
    use_hf_only=True,
    reduction="mean",
    loss_weight=0.1,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

iters = 60000
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type="OptimWrapper", optimizer=optimizer, clip_grad=None)

param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=iters, by_epoch=False)
]

train_cfg = dict(type="IterBasedTrainLoop", max_iters=iters, val_interval=iters // 10)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=iters // 10),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="SegVisualizationHook"),
)

custom_hooks = [
    dict(
        type="SDMTeacherEMAHook",
        momentum=0.999,
        update_buffers=True,   # BN running stats까지 EMA로 할지(권장 True)
        priority="NORMAL",
    )
]

randomness = dict(seed=304)
