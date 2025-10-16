# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from tabulate import tabulate
from transformers import AutoImageProcessor


def run_and_print_results(framework_model, compiled_model, inputs, dtype_override=None):
    """
    Runs inference using both a framework model and a compiled model on a list of input images,
    then prints the results in a formatted table.

    Args:
        framework_model: The original framework-based model.
        compiled_model: The compiled version of the model.
        inputs: A list of images to process and classify.
        dtype_override: Optional torch.dtype to override the input's dtype.
    """
    label_dict = framework_model.config.id2label
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

    results = []
    for i, image in enumerate(inputs):
        processed_inputs = processor(image, return_tensors="pt")["pixel_values"]

        # Apply dtype override if provided
        if dtype_override is not None:
            processed_inputs = processed_inputs.to(dtype_override)

        cpu_logits = framework_model(processed_inputs)[0]
        cpu_conf, cpu_idx = cpu_logits.softmax(-1).max(-1)
        cpu_pred = label_dict.get(cpu_idx.item(), "Unknown")

        tt_logits = compiled_model(processed_inputs)[0]
        tt_conf, tt_idx = tt_logits.softmax(-1).max(-1)
        tt_pred = label_dict.get(tt_idx.item(), "Unknown")

        results.append([i + 1, cpu_pred, cpu_conf.item(), tt_pred, tt_conf.item()])

    print(
        tabulate(
            results,
            headers=[
                "Example",
                "CPU Prediction",
                "CPU Confidence",
                "Compiled Prediction",
                "Compiled Confidence",
            ],
            tablefmt="grid",
        )
    )
