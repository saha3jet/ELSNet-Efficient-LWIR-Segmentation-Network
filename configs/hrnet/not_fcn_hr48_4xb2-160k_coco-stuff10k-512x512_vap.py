# Variation-Aware Proxy Learning (paper reimplementation)
# This config *inherits* a baseline config and only patches the decode head to a ProxyHead.
#
# Paper defaults:
#   K_c=5, tau=10, lambda_var=1, gamma=2, tau_R=0.8, lambda_r=1
# See "Variation-aware proxy learning for semantic segmentation" (Neurocomputing 2026).

_base_ = './fcn_hr48_4xb2-160k_coco-stuff10k-512x512.py'

# If you did not add these modules into mmseg's __init__.py exports,
# keep custom_imports so registry can find the new head/loss.
custom_imports = dict(
    imports=[
        'mmseg.models.decode_heads.proxy_heads',
        'mmseg.models.losses.variation_aware_proxy_loss',
    ],
    allow_failed_imports=False
)

model = dict(
    decode_head=dict(
        type='FCNProxyHead',
        # Proxy branch params
        embedding_dim=128,
        num_variations=5,
        proxy_upsample_first=False,
        # Compositional Similarity Loss (paper)
        loss_proxy=dict(
            type='VariationAwareProxyLoss',
            temperature=10.0,
            lambda_var=1.0,
            gamma=2.0,
            tau_R=0.8,
            lambda_r=1.0,
            loss_weight=1.0,          # paper uses λ_cs=1.0 by default
            softmax_mode='global',    # enables rep-proxy learning end-to-end
            class_chunk_size=32,
        ),
    ),
)
