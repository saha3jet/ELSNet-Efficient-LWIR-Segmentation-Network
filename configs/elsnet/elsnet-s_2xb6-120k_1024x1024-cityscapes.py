_base_ = [
    "../_base_/datasets/lwir_cityscapes_1024x1024.py",
    "../_base_/default_runtime.py",
]

class_weight = [
    0.8373,
    0.918,
    0.866,
    1.0345,
    1.0166,
    0.9969,
    0.9754,
    1.0489,
    0.8786,
    1.0023,
    0.9539,
    0.9843,
    1.1116,
    0.9037,
    1.0865,
    1.0955,
    1.0865,
    1.1529,
    1.0507,
]

crop_size = (1024, 1024)
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
        channels=32,
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
            direction="both",
        ),
    ),
    decode_head=dict(
        type="ELSHead",
        in_channels=128,
        channels=128,
        num_classes=19,
        norm_cfg=norm_cfg,
        act_cfg=dict(type="ReLU", inplace=True),
        align_corners=True,
        loss_decode=[
            dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                class_weight=class_weight,
                loss_weight=0.4,
            ),
            dict(
                type="OhemCrossEntropy",
                thres=0.9,
                min_kept=131072,
                class_weight=class_weight,
                loss_weight=1.0,
            ),
            dict(type="BoundaryLoss", loss_weight=20.0),
            dict(
                type="BoundarySemanticLoss",
                threshold=0.8,
                loss_weight=1.0,
                ignore_index=255,
            ),
        ],
    ),
    loss_imse=dict(
        type="IMSELoss", inverse=True, eps=1e-6, reduction="mean", loss_weight=0.1
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

iters = 120000
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

randomness = dict(seed=304)
