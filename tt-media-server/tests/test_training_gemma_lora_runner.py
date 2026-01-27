# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import torch
import os
from unittest.mock import MagicMock, patch, Mock
from tt_model_runners.forge_training_runners.training_gemma_lora_runner import TrainingGemmaLoraRunner
from utils.logger import TTLogger

class TestTrainingGemmaLoraRunner:
    @pytest.fixture
    def mock_training_request(self):
        """Creates a mock TrainingRequest object with necessary attributes."""
        request = MagicMock()
        request.batch_size = 4
        request.lora_r = 4
        request.lora_alpha = 8
        request.lora_target_modules = ["q_proj", "v_proj"]
        request.lora_task_type = "CAUSAL_LM"
        request.dtype = "torch.bfloat16"
        request.learning_rate = 1e-4
        request.ignored_index = -100
        request.num_epochs = 1
        request.val_steps_freq = 50
        request.steps_freq = 10
        return request

    @pytest.fixture
    def runner(self):
        """Initialize runner with mocked base class dependencies."""
        with patch("tt_model_runners.base_device_runner.get_settings"), \
             patch("tt_model_runners.base_device_runner.TTLogger"):
            return TrainingGemmaLoraRunner(device_id="test_device_0")

    @patch("tt_model_runners.training.training_gemma_runner.AutoModelForCausalLM")
    @patch("tt_model_runners.training.training_gemma_runner.get_dataset_loader")
    @patch("tt_model_runners.training.training_gemma_runner.xr")
    def test_warmup_configures_hardware_and_loading(self, mock_xr, mock_get_dataset, mock_hf, runner):
        """Verifies that warmup sets correct environment variables and loads components."""
        # Setup mocks
        mock_hf.from_pretrained.return_value = MagicMock()
        
        runner.warmup()

        # Check Environment Variables
        assert os.environ["PJRT_DEVICE"] == "TT"
        assert os.environ["XLA_STABLEHLO_COMPILE"] == "1"

        assert runner.device_id == "test_device_0"
        assert runner.logger is not None
        assert isinstance(runner.logger, TTLogger)
        
        # Check Hardware calls
        mock_xr.set_device_type.assert_called_with("TT")
        
        # Check Dataset calls
        assert mock_get_dataset.call_count == 2
        mock_get_dataset.assert_any_call(runner.model_name, split="train", collate_fn=pytest.any)

    @patch("tt_model_runners.training.training_gemma_runner.get_peft_model")
    @patch("tt_model_runners.training.training_gemma_runner.torch_xla.sync")
    def test_run_training_loop_execution(self, mock_sync, mock_get_peft, runner, mock_training_request):
        """Tests that the training loop processes batches and calls sync."""
        # Setup Mock Model
        mock_model = MagicMock()
        mock_get_peft.return_value = mock_model
        runner.hf_model = MagicMock()
        runner.device = "cpu" # Use CPU for unit test logic
        
        # Setup Mock Dataloader with one fake batch
        fake_batch = {
            "input_ids": torch.ones((1, 10), dtype=torch.long),
            "attention_mask": torch.ones((1, 10), dtype=torch.long),
            "labels": torch.ones((1, 10), dtype=torch.long)
        }
        mock_loader = MagicMock()
        mock_loader.__iter__.return_value = [fake_batch]
        
        runner.train_dataset = MagicMock()
        runner.train_dataset.get_dataloader.return_value = mock_loader
        runner.eval_dataset = MagicMock()
        runner.eval_dataset.get_dataloader.return_value = MagicMock()

        # Execute and assert
        runner.run([mock_training_request])

        assert mock_model.train.called
        mock_model.assert_called() # Ensure forward pass was called
        mock_sync.assert_called() # Ensure XLA sync was triggered after backward/step