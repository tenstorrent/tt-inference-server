# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Callbacks bridging tt-blacksmith's ``Trainer`` to the fine-tuning job API."""

import os
import time
import traceback

import torch_xla
from blacksmith.tools.trainer.callback import Callback


class StopTraining(Exception):
    """Raised from a callback to end the training loop early.

    ``Trainer._train_lifecycle`` catches whatever the loop raises and does not
    re-raise, so this is the way to leave the loop without failing the job.
    ``JobControlCallback.on_error`` tells it apart from a real failure.
    """


class JobCallback(Callback):
    """Base for callbacks that outlive the job they report on."""

    def __init__(self, logger):
        self._logger = logger
        self._request = None

    def bind(self, request) -> None:
        """Point the callback at a new job and drop the previous one's state."""
        self._request = request
        self._reset()

    def _reset(self) -> None:
        pass


class JobMetricsCallback(JobCallback):
    """Feeds train/validation losses to the job's metrics and logs.

    Progress (epoch/step, validation batch index) is logged every step so the
    job stream moves while tqdm stays on process stdout. Loss itself is only
    logged every ``metrics.steps_freq`` steps, plus once at the end of validation.
    """

    def _reset(self) -> None:
        self._running_loss = 0.0
        self._step_running_loss = 0.0
        self._prev_global_step = 0
        self._val_batch = 0
        self.last_train_loss = None
        self.last_val_loss = None

    def on_backward_end(self, trainer, loss, *args, **kwargs):
        self._step_running_loss += loss.item()

    def on_validation_start(self, trainer, *args, **kwargs):
        self._val_batch = 0
        # Surface the long first XLA/TT compile in job logs (tqdm stays on stdout).
        n_batches = None
        try:
            n_batches = len(trainer.val_dataloader)
        except TypeError:
            pass
        msg = (
            f"{self._epoch_step_label(trainer)} | Starting validation"
            + (f" ({n_batches} batches)" if n_batches is not None else "")
            + "; first batch may take several minutes to compile"
        )
        self._logger.info(msg, extra={"log_type": "info", "step": trainer.global_step})

    def on_validation_batch_end(self, trainer, batch, loss, *args, **kwargs):
        self._val_batch += 1
        total = None
        try:
            total = len(trainer.val_dataloader)
        except TypeError:
            pass
        suffix = f"/{total}" if total is not None else ""
        self._logger.info(
            f"{self._epoch_step_label(trainer)} | Validation batch {self._val_batch}{suffix}",
            extra={"log_type": "info", "step": trainer.global_step},
        )

    def on_train_batch_end(self, trainer, *args, **kwargs):
        # Under gradient accumulation most micro-batches do not advance
        # `global_step`; only finalize the ones that completed a step.
        if trainer.global_step == self._prev_global_step:
            return
        self._prev_global_step = trainer.global_step

        self._running_loss += (
            self._step_running_loss / trainer.config.gradient_accumulation_steps
        )
        self._step_running_loss = 0.0

        self._logger.info(
            self._epoch_step_label(trainer),
            extra={"log_type": "info", "step": trainer.global_step},
        )

        steps_freq = trainer.config.metrics.steps_freq
        if trainer.global_step % steps_freq:
            return

        self.last_train_loss = self._running_loss / steps_freq
        self._running_loss = 0.0
        self._record(trainer, "train_loss", self.last_train_loss, log_type="info")

    def on_validation_end(self, trainer, val_loss, *args, **kwargs):
        self.last_val_loss = val_loss
        self._record(trainer, "val_loss", val_loss, log_type="eval")

    @staticmethod
    def _epoch_step_label(trainer) -> str:
        num_epochs = getattr(trainer.config, "num_epochs", None)
        epoch = trainer.epoch + 1
        epoch_part = f"Epoch {epoch}/{num_epochs}" if num_epochs else f"Epoch {epoch}"
        return f"{epoch_part} | Step {trainer.global_step}"

    def _record(self, trainer, name: str, value: float, log_type: str) -> None:
        self._logger.info(
            f"{self._epoch_step_label(trainer)} | {name}: {value:.6f}",
            extra={"log_type": log_type, "step": trainer.global_step},
        )
        if self._request is None or self._request._training_metrics is None:
            return
        self._request._training_metrics.append(
            {
                "global_step": trainer.global_step,
                "epoch": trainer.epoch,
                "metric_name": name,
                "value": round(value, 6),
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "timestamp": time.time(),
            }
        )


class AdapterCheckpointCallback(JobCallback):
    """Saves the LoRA adapter every ``save_interval`` optimizer steps."""

    def __init__(self, logger, metrics: JobMetricsCallback):
        super().__init__(logger)
        self._metrics = metrics

    def _reset(self) -> None:
        self._prev_global_step = 0

    def on_train_batch_end(self, trainer, *args, **kwargs):
        save_interval = self._request.save_interval
        if save_interval <= 0 or trainer.global_step == self._prev_global_step:
            return
        self._prev_global_step = trainer.global_step

        if trainer.global_step % save_interval:
            return
        self._save(trainer)

    def _save(self, trainer) -> None:
        checkpoint_id = f"ckpt-step-{trainer.global_step}"
        checkpoint_path = os.path.join(self._request._output_model_path, checkpoint_id)
        try:
            # Filter to adapter tensors before moving to host: PeftModel.state_dict()
            # is the full base+adapter dict, so copying all of it would allocate a
            # second copy of the base weights on every save.
            state_dict = {
                key: value.cpu()
                for key, value in trainer.peft_model.state_dict().items()
                if "lora_" in key
            }
            trainer.peft_model.save_pretrained(checkpoint_path, state_dict=state_dict)
        except Exception as exception:
            self._logger.error(
                f"Failed to save checkpoint at step {trainer.global_step}: {exception}"
            )
            return

        self._logger.info(
            f"Model checkpoint saved to {checkpoint_path}.",
            extra={"log_type": "checkpoint", "step": trainer.global_step},
        )
        if self._request._training_checkpoints is None:
            return

        metrics = {}
        if self._metrics.last_train_loss is not None:
            metrics["train_loss"] = round(self._metrics.last_train_loss, 6)
        if self._metrics.last_val_loss is not None:
            metrics["val_loss"] = round(self._metrics.last_val_loss, 6)
        self._request._training_checkpoints.append(
            {
                "id": checkpoint_id,
                "step": trainer.global_step,
                "epoch": trainer.epoch + 1,
                "metrics": metrics,
                "created_at": time.time(),
            }
        )
        torch_xla.sync(wait=True)


class JobControlCallback(JobCallback):
    """Enforces ``max_steps`` and cancellation, and captures real failures."""

    def _reset(self) -> None:
        self.error = None
        self.stop_reason = None

    def on_train_batch_start(self, trainer, batch, *args, **kwargs):
        self._stop_if_cancelled(trainer)

    def on_validation_batch_start(self, trainer, batch, *args, **kwargs):
        self._stop_if_cancelled(trainer)

    def on_validation_batch_end(self, trainer, batch, loss, *args, **kwargs):
        self._stop_if_cancelled(trainer)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        request = self._request
        # Checked per micro-batch, so cancellation stays responsive even when
        # `global_step` only advances every `gradient_accumulation_steps` batches.
        self._stop_if_cancelled(trainer)
        if request.max_steps > 0 and trainer.global_step >= request.max_steps:
            self._stop(
                trainer,
                f"Reached max_steps={request.max_steps} at step {trainer.global_step}, "
                "stopping training.",
            )

    def _stop_if_cancelled(self, trainer) -> None:
        request = self._request
        if request is None or request._cancel_event is None:
            return
        if not request._cancel_event.is_set():
            return
        self._stop(
            trainer,
            f"Job cancelled at step {trainer.global_step}. "
            f"Directory containing checkpoints: {request._output_model_path}",
        )

    def _stop(self, trainer, reason: str) -> None:
        self.stop_reason = reason
        self._logger.info(
            reason, extra={"log_type": "info", "step": trainer.global_step}
        )
        raise StopTraining(reason)

    def on_error(self, trainer, exception, *args, **kwargs):
        if isinstance(exception, StopTraining):
            return
        self.error = exception
        self._logger.error(
            f"Job failed at step {trainer.global_step}: {exception}",
            extra={"log_type": "error", "step": trainer.global_step},
        )
        self._logger.error(
            f"Full traceback: {traceback.format_exc()}",
            extra={"log_type": "error", "step": trainer.global_step},
        )
