from copy import deepcopy
from collections import OrderedDict
from lightning.pytorch.callbacks.callback import Callback
from torch.optim.swa_utils import AveragedModel


class ModelAveragingCallback(Callback):
    def __init__(self, device, avg_fn, update_after_steps=-1):
        self._device = device
        self._avg_fn = avg_fn
        self._averaged_model = None
        self._latest_update_step = update_after_steps

    def on_fit_start(self, trainer, pl_module) -> None:
        device = self._device or pl_module.device
        self._averaged_model = AveragedModel(model=pl_module, device=device, avg_fn=self._avg_fn, use_buffers=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step > self._latest_update_step:
            self._averaged_model.update_parameters(pl_module)
            self._latest_update_step = trainer.global_step

    def on_fit_end(self, trainer, pl_module):
        self._copy_average_to_current(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._averaged_model is not None:
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._averaged_model is not None:
            self._swap_models(pl_module)

    def state_dict(self):
        return {"latest_update_step": self._latest_update_step}

    def load_state_dict(self, state_dict):
        self._latest_update_step = state_dict["latest_update_step"]

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        average_state = self._averaged_model.state_dict()
        checkpoint["current_state_dict"] = checkpoint["state_dict"]
        checkpoint["state_dict"] = OrderedDict({
            name[7:]: value for name, value in average_state.items() if name.startswith("module.")
        })
        # checkpoint["model_averaging_state"] = {
        #     name: value for name, value in average_state.items() if not name.startswith("module.")
        # }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if ("current_state_dict" in checkpoint) and ("model_averaging_state" in checkpoint):
            average_state = {"module." + name: value for name, value in checkpoint["state_dict"].items()}
            average_state |= checkpoint["model_averaging_state"]
            self._averaged_model.load_state_dict(average_state)
            checkpoint["state_dict"] = checkpoint["current_state_dict"]
        else:
            self._averaged_model.module.load_state_dict(deepcopy(checkpoint['state_dict']), strict=False)

    def _swap_models(self, pl_module):
        average_tensors, current_tensors = self._get_named_tensors(pl_module)
        for name, average_tensor in average_tensors.items():
            current_tensor = current_tensors[name]
            tmp = average_tensor.data.clone()
            average_tensor.data.copy_(current_tensor.data)
            current_tensor.data.copy_(tmp)

    def _copy_average_to_current(self, pl_module):
        average_tensors, current_tensors = self._get_named_tensors(pl_module)
        for name, average_tensor in average_tensors.items():
            current_tensors[name].data.copy_(average_tensor.data)

    def _get_named_tensors(self, pl_module):
        average_tensors = self._averaged_model.module.state_dict(keep_vars=True)
        current_tensors = pl_module.state_dict(keep_vars=True)

        average_keys = set(average_tensors.keys())
        current_keys = set(current_tensors.keys())
        if average_keys != current_keys:
            missing_in_current = sorted(average_keys - current_keys)
            missing_in_average = sorted(current_keys - average_keys)
            raise RuntimeError(
                "EMA model tensors do not match the current model. "
                f"Missing in current: {missing_in_current[:5]}; "
                f"missing in averaged: {missing_in_average[:5]}"
            )

        for name in average_tensors:
            if average_tensors[name].shape != current_tensors[name].shape:
                raise RuntimeError(
                    "EMA tensor shape mismatch for "
                    f"'{name}': averaged {tuple(average_tensors[name].shape)} vs "
                    f"current {tuple(current_tensors[name].shape)}"
                )

        return average_tensors, current_tensors


class SaveConfigCallback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # 保存 cfg 到 checkpoint 字典中
        checkpoint['cfg'] = self.cfg
