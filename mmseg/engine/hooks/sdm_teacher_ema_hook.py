from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class SDMTeacherEMAHook(Hook):
    def __init__(self, momentum=0.999, update_buffers=True, priority="NORMAL"):
        self.momentum = float(momentum)
        self.update_buffers = bool(update_buffers)
        self.priority = priority  # (MMEngine은 priority를 config에서 처리하지만, 넣어둬도 무해)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model
        if hasattr(model, "module"):  # DDP
            model = model.module

        backbone = getattr(model, "backbone", None)
        if backbone is None:
            return

        if hasattr(backbone, "update_sdm_teacher"):
            backbone.update_sdm_teacher(
                momentum=self.momentum,
                update_buffers=self.update_buffers,
            )
