# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Callbacks bridging ttml's ``Trainer`` to the fine-tuning job API."""

import os
import time
import traceback
from datetime import timedelta

from ttml.unified_trainer import Callback


class StopTraining(Exception):
    """Raised from a callback to end the training loop early.

    Like any exception, it goes through ``on_train_error`` and propagates out of
    ``Trainer.train()``. The runner treats it as a successful end of the job, and
    the job callbacks' ``on_train_error`` tell it apart from a real failure.
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

    def _touch_progress(self) -> None:
        if self._request is not None:
            self._request.touch_progress()


class JobMetricsCallback(JobCallback):
    """Feeds train/validation losses to the job's metrics and logs.

    Progress (epoch/step, validation batch index) is logged every step so the
    job stream moves while tqdm stays on process stdout. A job's first train
    forward, train backward, optimizer and validation passes are announced
    before they run, since compiling the kernels they use for the first time
    takes minutes, and the end of the loop is logged with its duration since
    ``on_train_start``.
    Loss itself is only logged every ``metrics.steps_freq`` steps, plus once at
    the end of validation.
    """

    def _reset(self) -> None:
        self._running_loss = None
        self._val_batch = 0
        self._compile_announced = set()
        self._train_started_at = None
        self.last_train_loss = None
        self.last_val_loss = None

    def on_optimizer_step_end(self, trainer, window_loss, *args, **kwargs):
        # window_loss is the mean loss of the step's micro-batches. The logging
        # window's sum is read with `loss_to_float` once, when it is logged.
        if self._running_loss is None:
            self._running_loss = window_loss
        else:
            self._running_loss = self._running_loss + window_loss

    def on_train_start(self, trainer, *args, **kwargs):
        self._train_started_at = time.monotonic()
        n_batches = None
        try:
            n_batches = len(trainer.train_dataloader)
        except TypeError:
            pass
        msg = f"{self._epoch_step_label(trainer)} | Starting training" + (
            f" ({n_batches} batches per epoch)" if n_batches is not None else ""
        )
        self._logger.info(msg, extra={"log_type": "info", "step": trainer.global_step})

    def on_forward_start(self, trainer, batch, *args, **kwargs):
        self._announce_compile(trainer, "train forward")

    def on_backward_start(self, trainer, loss, *args, **kwargs):
        self._announce_compile(trainer, "train backward")

    def on_optimizer_step_start(self, trainer, *args, **kwargs):
        self._announce_compile(trainer, "optimizer")

    def on_validation_start(self, trainer, *args, **kwargs):
        self._val_batch = 0
        n_batches = None
        try:
            n_batches = len(trainer.val_dataloader)
        except TypeError:
            pass
        msg = f"{self._epoch_step_label(trainer)} | Starting validation" + (
            f" ({n_batches} batches)" if n_batches is not None else ""
        )
        self._logger.info(msg, extra={"log_type": "info", "step": trainer.global_step})

    def on_validation_batch_start(self, trainer, batch, *args, **kwargs):
        # Validation has no forward hook; its forward runs right after this one.
        self._announce_compile(trainer, "validation")

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

    def on_step_end(self, trainer, global_step, *args, **kwargs):
        self._logger.info(
            self._epoch_step_label(trainer),
            extra={"log_type": "info", "step": global_step},
        )

        steps_freq = trainer.config.metrics.steps_freq
        if global_step % steps_freq:
            return

        self.last_train_loss = trainer.loss_to_float(self._running_loss) / steps_freq
        self._running_loss = None
        self._record(trainer, "train_loss", self.last_train_loss, log_type="info")

    def on_validation_end(self, trainer, val_loss, *args, **kwargs):
        self.last_val_loss = trainer.loss_to_float(val_loss)
        self._record(trainer, "val_loss", self.last_val_loss, log_type="eval")

    def on_train_end(self, trainer, *args, **kwargs):
        self._log_training_end(trainer, cancelled=False)

    def on_train_error(self, trainer, exception, *args, **kwargs):
        # max_steps and cancellation end the loop with StopTraining, so on_train_end never fires.
        if not isinstance(exception, StopTraining):
            return
        cancel_event = getattr(self._request, "_cancel_event", None)
        cancelled = cancel_event is not None and cancel_event.is_set()
        self._log_training_end(trainer, cancelled=cancelled)

    @staticmethod
    def _epoch_step_label(trainer) -> str:
        num_epochs = getattr(trainer.config, "num_epochs", None)
        epoch = trainer.epoch + 1
        epoch_part = f"Epoch {epoch}/{num_epochs}" if num_epochs else f"Epoch {epoch}"
        return f"{epoch_part} | Step {trainer.global_step}"

    def _announce_compile(self, trainer, pass_name: str) -> None:
        if pass_name in self._compile_announced:
            return
        self._compile_announced.add(pass_name)
        self._logger.info(
            f"{self._epoch_step_label(trainer)} | First {pass_name} pass; compiling "
            "its kernels on first use can take several minutes...",
            extra={"log_type": "info", "step": trainer.global_step},
        )

    def _log_training_end(self, trainer, cancelled: bool) -> None:
        # Raising from on_train_error would replace StopTraining and fail the job.
        if self._train_started_at is None:
            return
        elapsed = timedelta(seconds=round(time.monotonic() - self._train_started_at))
        outcome = (
            f"cancelled after {elapsed}" if cancelled else f"finished in {elapsed}"
        )
        self._logger.info(
            f"{self._epoch_step_label(trainer)} | Training {outcome}",
            extra={"log_type": "info", "step": trainer.global_step},
        )

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
                # TODO: record every optimizer's learning rate once the metrics
                # store has a column for several; jobs train with one optimizer.
                "learning_rate": trainer.current_learning_rate()[0],
                "timestamp": time.time(),
            }
        )


class AdapterCheckpointCallback(JobCallback):
    """Saves the LoRA adapter every ``checkpoint.steps_freq`` optimizer steps."""

    def __init__(self, logger, metrics: JobMetricsCallback):
        super().__init__(logger)
        self._metrics = metrics

    def on_step_end(self, trainer, global_step, *args, **kwargs):
        config = trainer.config.checkpoint
        if config.save_strategy != "step":
            return
        if global_step % config.steps_freq:
            return
        self._save(trainer)

    def _save(self, trainer) -> None:
        checkpoint_id = f"ckpt-step-{trainer.global_step}"
        checkpoint_path = os.path.join(
            trainer.config.checkpoint.project_dir, checkpoint_id
        )
        try:
            trainer.save_checkpoint(
                checkpoint_path,
                {
                    "step": trainer.global_step,
                    "epoch": trainer.epoch,
                    "metrics": self._checkpoint_metrics(),
                },
            )
        except Exception as exception:
            self._logger.error(
                f"Failed to save checkpoint at step {trainer.global_step}: {exception}"
            )
            return

        self._logger.info(
            f"Model checkpoint saved to {checkpoint_path}.",
            extra={"log_type": "checkpoint", "step": trainer.global_step},
        )
        if self._request is None or self._request._training_checkpoints is None:
            return

        self._request._training_checkpoints.append(
            {
                "id": checkpoint_id,
                "step": trainer.global_step,
                "epoch": trainer.epoch + 1,
                "metrics": self._checkpoint_metrics(),
                "created_at": time.time(),
            }
        )

    def _checkpoint_metrics(self) -> dict:
        metrics = {}
        if self._metrics.last_train_loss is not None:
            metrics["train_loss"] = round(self._metrics.last_train_loss, 6)
        if self._metrics.last_val_loss is not None:
            metrics["val_loss"] = round(self._metrics.last_val_loss, 6)
        return metrics


class JobControlCallback(JobCallback):
    """Enforces ``max_steps`` and cancellation, and logs real failures to the job."""

    def _reset(self) -> None:
        self.stop_reason = None

    def on_train_start(self, trainer, *args, **kwargs):
        self._touch_progress()

    def on_validation_start(self, trainer, *args, **kwargs):
        self._touch_progress()

    def on_train_batch_start(self, trainer, batch, *args, **kwargs):
        self._stop_if_cancelled(trainer)

    def on_validation_batch_start(self, trainer, batch, *args, **kwargs):
        self._stop_if_cancelled(trainer)

    def on_validation_batch_end(self, trainer, batch, loss, *args, **kwargs):
        self._touch_progress()
        self._stop_if_cancelled(trainer)

    def on_train_batch_end(self, trainer, *args, **kwargs):
        self._touch_progress()
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

    def on_setup_error(self, trainer, exception, *args, **kwargs):
        self._log_failure(trainer, f"Job failed during setup: {exception}")

    def on_train_error(self, trainer, exception, *args, **kwargs):
        if isinstance(exception, StopTraining):
            return
        self._log_failure(
            trainer, f"Job failed at step {trainer.global_step}: {exception}"
        )

    def _log_failure(self, trainer, message: str) -> None:
        self._logger.error(
            message, extra={"log_type": "error", "step": trainer.global_step}
        )
        self._logger.error(
            f"Full traceback: {traceback.format_exc()}",
            extra={"log_type": "error", "step": trainer.global_step},
        )
