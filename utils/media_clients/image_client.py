# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import aiohttp
import requests

from .base_strategy_interface import BaseMediaStrategy
from .test_status import ImageGenerationTestStatus

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.sdxl_accuracy_utils.sdxl_accuracy_utils import (
    calculate_accuracy_check,
    calculate_metrics,
    sdxl_get_prompts,
)
from workflows.utils import get_num_calls, is_sdxl_num_prompts_enabled

logger = logging.getLogger(__name__)

# SDXL specific constants
WORKFLOW_EVALS = "evals"
WORKFLOW_BENCHMARKS = "benchmarks"
SDXL_SD35_BENCHMARK_NUM_PROMPTS = 20
SDXL_SD35_INFERENCE_STEPS = 20
IMAGE_FORMAT_FOR_EVALS = "PNG"
IMAGE_QUALITY_FOR_EVALS = 100
SDXL_INPAINTING_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = (
    "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
)
GUIDANCE_SCALE = 8
NUM_INFERENCE_STEPS = 20

# IMG2IMG specific constants
SDXL_IMG2IMG_INFERENCE_STEPS = 30
GUIDANCE_SCALE_IMG2IMG = 7.5
SEED_IMG2IMG = 0
STRENGTH_IMG2IMG = 0.6

# INPAINTING specific constants
GUIDANCE_SCALE_INPAINTING = 8.0
SEED_INPAINTING = 0
STRENGTH_INPAINTING = 0.99


class ImageClientStrategy(BaseMediaStrategy):
    """Strategy for image models (SDXL, etc)."""

    def __init__(self, all_params, model_spec, device, output_path, service_port):
        super().__init__(all_params, model_spec, device, output_path, service_port)

        # Map runners to their benchmark methods
        self.benchmark_methods = {
            "tt-sdxl-trace": self._run_image_generation_benchmark,
            "tt-sdxl-image-to-image": self._run_img2img_generation_benchmark,
            "tt-sdxl-edit": self._run_inpainting_generation_benchmark,
            "tt-sd3.5": self._run_image_generation_benchmark,
        }

        # Map runners to their eval methods (for future use)
        self.eval_methods = {
            "tt-sdxl-trace": self._run_image_generation_eval,
            "tt-sdxl-image-to-image": self._run_img2img_generation_eval,
            "tt-sdxl-edit": self._run_inpainting_generation_eval,
            "tt-sd3.5": self._run_image_generation_eval,
        }

    def run_eval(self) -> None:
        """Run evaluations for the model."""
        status_list = []

        logger.info(
            f"Running evals for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            # Route to appropriate eval method using dispatch map
            eval_method = self.eval_methods.get(
                runner_in_use, self._run_image_generation_eval
            )
            status_list, total_time = asyncio.run(eval_method())
        except Exception as e:
            logger.error(f"Eval execution encountered an error: {e}")
            raise

        logger.info("Generating eval report...")
        benchmark_data = {}

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)
        logger.info(f"Extracted TTFT value: {ttft_value}")

        benchmark_data["model"] = self.model_spec.model_name
        benchmark_data["device"] = self.device.name.lower()
        benchmark_data["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        )
        benchmark_data["task_type"] = "image"
        benchmark_data["task_name"] = self.all_params.tasks[0].task_name
        benchmark_data["tolerance"] = self.all_params.tasks[0].score.tolerance
        benchmark_data["published_score"] = self.all_params.tasks[
            0
        ].score.published_score
        benchmark_data["score"] = ttft_value
        benchmark_data["published_score_ref"] = self.all_params.tasks[
            0
        ].score.published_score_ref

        logger.info("Running and calculating accuracy and metrics")
        fid_score, average_clip_score, deviation_clip_score = calculate_metrics(
            status_list
        )
        accuracy_check = calculate_accuracy_check(
            fid_score,
            average_clip_score,
            len(status_list),
            self.model_spec.model_name,
        )

        benchmark_data["fid_score"] = fid_score
        benchmark_data["average_clip"] = average_clip_score
        benchmark_data["deviation_clip_score"] = deviation_clip_score
        benchmark_data["accuracy_check"] = accuracy_check

        # Calculate tput_user for tt-sd models only
        device_spec = self.model_spec.device_model_spec
        if device_spec and hasattr(device_spec, "max_concurrency"):
            tput_user = len(status_list) / (total_time * device_spec.max_concurrency)
            benchmark_data["tput_user"] = tput_user
            logger.info(
                f"Calculated tput_user: {tput_user} (prompts: {len(status_list)}, time: {total_time}s, max_concurrency: {device_spec.max_concurrency})"
            )
        else:
            logger.warning(f"No device spec found for device: {self.device}")

        # Make benchmark_data is inside of list as an object
        benchmark_data = [benchmark_data]

        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / self.model_spec.hf_model_repo.replace("/", "__")
            / f"results_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)

        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        logger.info(f"Evaluation data written to: {eval_filename}")

    def run_benchmark(self, attempt=0) -> list[ImageGenerationTestStatus]:
        """Run benchmarks for the model."""
        logger.info(
            f"Running benchmarks for model: {self.model_spec.model_name} on device: {self.device.name}"
        )
        try:
            health_status, runner_in_use = self.get_health()
            if health_status:
                logger.info("Health check passed.")
            else:
                logger.error("Health check failed.")
                raise

            logger.info(f"Runner in use: {runner_in_use}")

            # Get num_calls from benchmark parameters
            num_calls = get_num_calls(self)

            # Route to appropriate benchmark method using dispatch map
            benchmark_method = self.benchmark_methods.get(
                runner_in_use, self._run_image_generation_benchmark
            )
            status_list = benchmark_method(num_calls)

            self._generate_report(status_list)
        except Exception as e:
            logger.error(f"Benchmark execution encountered an error: {e}")
            raise

    def _calculate_ttft_value(
        self, status_list: list[ImageGenerationTestStatus]
    ) -> float:
        """Calculate TTFT value based on model type and status list."""
        logger.info("Calculating TTFT value")

        return (
            sum(status.elapsed for status in status_list) / len(status_list)
            if status_list
            else 0
        )

    def _generate_report(
        self,
        status_list: list[ImageGenerationTestStatus],
    ) -> None:
        """Generate benchmark report."""
        logger.info("Generating benchmark report...")
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time.time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        # Calculate TTFT
        ttft_value = self._calculate_ttft_value(status_list)

        # Convert ImageGenerationTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "num_inference_steps": status_list[0].num_inference_steps
                if status_list
                else 0,
                "ttft": ttft_value,
                "inference_steps_per_second": sum(
                    status.inference_steps_per_second for status in status_list
                )
                / len(status_list)
                if status_list
                else 0,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task_type": "image",
        }

        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Report generated: {result_filename}")

    async def _run_image_generation_eval(
        self,
    ) -> tuple[list[ImageGenerationTestStatus], float]:
        """Run image generation evals."""
        logger.info("Running image generation eval.")

        num_prompts = is_sdxl_num_prompts_enabled(self)
        logger.info(f"Number of prompts set to: {num_prompts}")

        prompts = sdxl_get_prompts(0, num_prompts)
        logger.info(f"Retrieved {len(prompts)} prompts for evaluation.")

        # Create all images concurrently
        async with aiohttp.ClientSession() as session:
            total_start_time = time.time()
            tasks = [
                self._generate_image_eval_async(session, prompt) for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - total_start_time

        logger.info(
            f"Generated {len(prompts)} images concurrently in {total_time:.2f} seconds"
        )

        # Process results into ImageGenerationTestStatus objects and filter out failed generations
        status_list = []
        failed_count = 0

        for i, (status, elapsed, base64image) in enumerate(results):
            prompt = prompts[i]  # Get the corresponding prompt

            # Skip failed image generations
            if not status or base64image is None:
                failed_count += 1
                logger.warning(
                    f"❌ Skipping failed image {i + 1}/{num_prompts}: '{prompt}'"
                )
                continue

            inference_steps_per_second = (
                SDXL_SD35_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(f"🚀 Image {i + 1}/{num_prompts}: {prompt} - {elapsed:.2f}s")

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_SD35_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                    base64image=base64image,
                    prompt=prompt,
                )
            )

        logger.info(f"Total image generations attempted: {num_prompts}")
        logger.info(f"Total failed image generations: {failed_count}")
        logger.info(f"Total successful image generations: {num_prompts - failed_count}")

        if failed_count:
            logger.warning(f"⚠️  {failed_count} image generations failed during eval.")
            raise RuntimeError(
                f"❌ {failed_count} image generations failed - cannot calculate accuracy metrics"
            )

        return status_list, total_time

    async def _generate_image_eval_async(
        self, session: aiohttp.ClientSession, prompt: str
    ) -> tuple[bool, float, Optional[str]]:
        """Generate image using SDXL model with shared session. This is specific for evals workflow."""
        logger.info(f"🌅 Generating image for prompt: {prompt}")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": 0,
            "guidance_scale": GUIDANCE_SCALE,
            "image_return_format": IMAGE_FORMAT_FOR_EVALS,
            "image_quality": IMAGE_QUALITY_FOR_EVALS,
            "number_of_images": 1,
        }

        start_time = time.time()

        try:
            async with session.post(
                f"{self.base_url}/image/generations",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=25000),
            ) as response:
                elapsed = time.time() - start_time

                if response.status != 200:
                    logger.error(
                        f"❌ Image generation for eval failed with status: {response.status}"
                    )
                    return False, elapsed, None

                response_data = await response.json()
                images = response_data.get("images", [])
                base64image = images[0] if images else None

                logger.info(f"✅ Image generation for eval succeeded in {elapsed:.2f}s")
                return True, elapsed, base64image

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Image generation for eval failed: {e}")
            return False, elapsed, None

    async def _run_img2img_generation_eval(
        self,
    ) -> tuple[list[ImageGenerationTestStatus], float]:
        """Run image2image generation evals."""
        logger.info("Running image2image generation eval.")

        # Using a fixed prompt for img2img evals
        prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
        logger.info(f"Using 1 prompt for evaluation: {prompt}")

        # Load test image payload from file
        image_payload_path = f"{self.test_payloads_path}/image_client_img2img_payload"
        with open(image_payload_path, "r") as f:
            image_data = json.load(f)

        # Create image
        async with aiohttp.ClientSession() as session:
            total_start_time = time.time()
            tasks = [
                self._generate_image_img2img_eval_async(session, prompt, image_data)
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - total_start_time

        logger.info(f"Generated 1 img2img image in {total_time:.2f} seconds")

        # Process results into ImageGenerationTestStatus objects
        status_list = []
        failed_count = 0

        for i, (status, elapsed, base64image) in enumerate(results):
            # Skip failed image generations
            if not status or base64image is None:
                failed_count += 1
                logger.warning(f"❌ Failed img2img image generation: '{prompt}'")
                continue

            inference_steps_per_second = (
                SDXL_IMG2IMG_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(f"🚀 Img2img image: {prompt} - {elapsed:.2f}s")

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_IMG2IMG_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                    base64image=base64image,
                    prompt=prompt,
                )
            )

        logger.info("Total img2img generations attempted: 1")
        logger.info(f"Total failed img2img generations: {failed_count}")
        logger.info(f"Total successful img2img generations: {1 - failed_count}")

        if failed_count:
            logger.warning("⚠️  Img2img generation failed during eval.")
            raise RuntimeError(
                "❌ Img2img generation failed - cannot calculate accuracy metrics"
            )

        return status_list, total_time

    async def _generate_image_img2img_eval_async(
        self, session: aiohttp.ClientSession, prompt: str, image_data: dict
    ) -> tuple[bool, float, Optional[str]]:
        """Generate image using img2img model with shared session. This is specific for evals workflow."""
        logger.info(f"🌆 Generating img2img image for prompt: {prompt}")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image": image_data["file"],
            "seed": SEED_IMG2IMG,
            "guidance_scale": GUIDANCE_SCALE_IMG2IMG,
            "number_of_images": 1,
            "strength": STRENGTH_IMG2IMG,
            "num_inference_steps": SDXL_IMG2IMG_INFERENCE_STEPS,
        }
        start_time = time.time()

        try:
            async with session.post(
                f"{self.base_url}/image/image-to-image",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=25000),
            ) as response:
                elapsed = time.time() - start_time

                if response.status != 200:
                    logger.error(
                        f"❌ Img2img generation for eval failed with status: {response.status}"
                    )
                    return False, elapsed, None

                response_data = await response.json()
                images = response_data.get("images", [])
                base64image = images[0] if images else None

                logger.info(
                    f"✅ Img2img generation for eval succeeded in {elapsed:.2f}s"
                )
                return True, elapsed, base64image

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Img2img generation for eval failed: {e}")
            return False, elapsed, None

    async def _run_inpainting_generation_eval(
        self,
    ) -> tuple[list[ImageGenerationTestStatus], float]:
        """Run inpainting generation evals."""
        logger.info("Running inpainting generation eval.")

        # Using a fixed prompt for inpainting evals
        prompt = "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k"
        logger.info(f"Using 1 prompt for evaluation: {prompt}")

        # Load inpaint image and mask payload from file
        image_payload_path = (
            f"{self.test_payloads_path}/image_client_inpainting_payload"
        )
        with open(image_payload_path, "r") as f:
            payload_data = json.load(f)
            inpaint_image = payload_data["inpaint_image"]
            inpaint_mask = payload_data["inpaint_mask"]

        # Create image
        async with aiohttp.ClientSession() as session:
            total_start_time = time.time()
            tasks = [
                self._generate_image_inpainting_eval_async(
                    session, prompt, inpaint_image, inpaint_mask
                )
            ]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - total_start_time

        logger.info(f"Generated 1 inpainting image in {total_time:.2f} seconds")

        # Process results into ImageGenerationTestStatus objects
        status_list = []
        failed_count = 0

        for i, (status, elapsed, base64image) in enumerate(results):
            # Skip failed image generations
            if not status or base64image is None:
                failed_count += 1
                logger.warning(f"❌ Failed inpainting image generation: '{prompt}'")
                continue

            inference_steps_per_second = (
                SDXL_INPAINTING_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(f"🚀 Inpainting image: {prompt} - {elapsed:.2f}s")

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_INPAINTING_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                    base64image=base64image,
                    prompt=prompt,
                )
            )

        logger.info("Total inpainting generations attempted: 1")
        logger.info(f"Total failed inpainting generations: {failed_count}")
        logger.info(f"Total successful inpainting generations: {1 - failed_count}")

        if failed_count:
            logger.warning("⚠️  Inpainting generation failed during eval.")
            raise RuntimeError(
                "❌ Inpainting generation failed - cannot calculate accuracy metrics"
            )

        return status_list, total_time

    async def _generate_image_inpainting_eval_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        inpaint_image: dict,
        inpaint_mask: dict,
    ) -> tuple[bool, float, Optional[str]]:
        """Generate image using inpainting model with shared session. This is specific for evals workflow."""
        logger.info(f"🏞️ Generating inpainting image for prompt: {prompt}")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
            "image": inpaint_image,
            "mask": inpaint_mask,
            "seed": SEED_INPAINTING,
            "guidance_scale": GUIDANCE_SCALE_INPAINTING,
            "number_of_images": 1,
            "strength": STRENGTH_INPAINTING,
            "num_inference_steps": SDXL_INPAINTING_INFERENCE_STEPS,
        }

        start_time = time.time()

        try:
            async with session.post(
                f"{self.base_url}/image/edits",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=25000),
            ) as response:
                elapsed = time.time() - start_time

                if response.status != 200:
                    logger.error(
                        f"❌ Inpainting generation for eval failed with status: {response.status}"
                    )
                    return False, elapsed, None

                response_data = await response.json()
                images = response_data.get("images", [])
                base64image = images[0] if images else None

                logger.info(
                    f"✅ Inpainting generation for eval succeeded in {elapsed:.2f}s"
                )
                return True, elapsed, base64image

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ Inpainting generation for eval failed: {e}")
            return False, elapsed, None

    def _run_image_generation_benchmark(
        self, num_calls: int
    ) -> list[ImageGenerationTestStatus]:
        """Run image generation benchmark."""
        logger.info("Running image generation benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Generating image {i + 1}/{num_calls}...")
            status, elapsed = self._generate_image()
            inference_steps_per_second = (
                SDXL_SD35_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(
                f"Generated image with {SDXL_SD35_INFERENCE_STEPS} steps in {elapsed:.2f} seconds."
            )

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_SD35_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                )
            )

        return status_list

    def _generate_image(self, num_inference_steps: int = 20) -> tuple[bool, float]:
        """Generate image using SDXL model."""
        logger.info("🌅 Generating image")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": "Rabbit",
            "seed": 0,
            "guidance_scale": 3.0,
            "number_of_images": 1,
            "num_inference_steps": num_inference_steps,
        }
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/image/generations",
            json=payload,
            headers=headers,
            timeout=90,
        )
        elapsed = time.time() - start_time

        if response.status_code != 200:
            logger.error(
                f"❌ Image generation failed with status {response.status_code}"
            )
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except Exception as e:
                logger.error(f"Could not parse error response: {e}")
                logger.error(f"Raw response: {response.text[:500]}")
            raise RuntimeError(
                f"Image generation failed with status {response.status_code}"
            )

        logger.info(f"✅ Image generation successful in {elapsed:.2f}s")
        return (response.status_code == 200), elapsed

    def _run_img2img_generation_benchmark(
        self, num_calls: int
    ) -> list[ImageGenerationTestStatus]:
        """Run image-to-image generation benchmark."""
        logger.info("Running image-to-image generation benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Generating image {i + 1}/{num_calls}...")
            status, elapsed = self._generate_image_img2img()
            inference_steps_per_second = (
                SDXL_IMG2IMG_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(
                f"Generated image with {SDXL_IMG2IMG_INFERENCE_STEPS} steps in {elapsed:.2f} seconds."
            )

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_IMG2IMG_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                )
            )

        return status_list

    def _generate_image_img2img(
        self, num_inference_steps: int = SDXL_IMG2IMG_INFERENCE_STEPS
    ) -> tuple[bool, float]:
        """Generate image using img2img model."""
        logger.info("🌆 Generating image with img2img")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }

        # Load test image payload from file
        image_payload_path = f"{self.test_payloads_path}/image_client_img2img_payload"
        with open(image_payload_path, "r") as f:
            image_data = json.load(f)

        payload = {
            "prompt": "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
            "image": image_data["file"],
            "seed": SEED_IMG2IMG,
            "guidance_scale": GUIDANCE_SCALE_IMG2IMG,
            "number_of_images": 1,
            "strength": STRENGTH_IMG2IMG,
            "num_inference_steps": num_inference_steps,
        }
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/image/image-to-image",
            json=payload,
            headers=headers,
            timeout=90,
        )
        elapsed = time.time() - start_time

        if response.status_code != 200:
            logger.error(
                f"❌ Image-to-image generation failed with status {response.status_code}"
            )
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except Exception as e:
                logger.error(f"Could not parse error response: {e}")
                logger.error(f"Raw response: {response.text[:500]}")
            raise RuntimeError(
                f"Image-to-image generation failed with status {response.status_code}"
            )

        logger.info(f"✅ Image-to-image generation successful in {elapsed:.2f}s")
        return (response.status_code == 200), elapsed

    def _run_inpainting_generation_benchmark(
        self, num_calls: int
    ) -> list[ImageGenerationTestStatus]:
        """Run inpainting generation benchmark."""
        logger.info("Running inpainting generation benchmark.")
        status_list = []

        for i in range(num_calls):
            logger.info(f"Generating image {i + 1}/{num_calls}...")
            status, elapsed = self._generate_image_inpainting()
            inference_steps_per_second = (
                SDXL_INPAINTING_INFERENCE_STEPS / elapsed if elapsed > 0 else 0
            )
            logger.info(
                f"Generated image with {SDXL_INPAINTING_INFERENCE_STEPS} steps in {elapsed:.2f} seconds."
            )

            status_list.append(
                ImageGenerationTestStatus(
                    status=status,
                    elapsed=elapsed,
                    num_inference_steps=SDXL_INPAINTING_INFERENCE_STEPS,
                    inference_steps_per_second=inference_steps_per_second,
                )
            )

        return status_list

    def _generate_image_inpainting(
        self, num_inference_steps: int = SDXL_INPAINTING_INFERENCE_STEPS
    ) -> tuple[bool, float]:
        """Generate image using inpainting model."""
        logger.info("🏞️ Generating image with inpainting")
        headers = {
            "accept": "application/json",
            "Authorization": "Bearer your-secret-key",
            "Content-Type": "application/json",
        }

        # Load inpaint image and inpaint mask payload from file
        image_payload_path = (
            f"{self.test_payloads_path}/image_client_inpainting_payload"
        )
        with open(image_payload_path, "r") as f:
            payload_data = json.load(f)
            inpaint_image = payload_data["inpaint_image"]
            inpaint_mask = payload_data["inpaint_mask"]

        payload = {
            "prompt": "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k",
            "image": inpaint_image,
            "mask": inpaint_mask,
            "seed": SEED_INPAINTING,
            "guidance_scale": GUIDANCE_SCALE_INPAINTING,
            "number_of_images": 1,
            "strength": STRENGTH_INPAINTING,
            "num_inference_steps": num_inference_steps,
        }
        start_time = time.time()
        response = requests.post(
            f"{self.base_url}/image/edits",
            json=payload,
            headers=headers,
            timeout=90,
        )
        elapsed = time.time() - start_time

        if response.status_code != 200:
            logger.error(
                f"❌ Inpainting generation failed with status {response.status_code}"
            )
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
            except Exception as e:
                logger.error(f"Could not parse error response: {e}")
                logger.error(f"Raw response: {response.text[:500]}")
            raise RuntimeError(
                f"Inpainting generation failed with status {response.status_code}"
            )

        logger.info(f"✅ Inpainting generation successful in {elapsed:.2f}s")
        return (response.status_code == 200), elapsed
