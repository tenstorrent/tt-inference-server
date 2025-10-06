# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from time import time
import time as time_module
from typing import Optional


class SDXLTestStatus:
    status: bool
    elapsed: float
    num_inference_steps: Optional[int]
    inference_steps_per_second: Optional[float]

    def __init__(self, status: bool, elapsed: float, num_inference_steps: int = 0, inference_steps_per_second: float = 0):
        self.status = status
        self.elapsed = elapsed
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second


class ImageClient:
    def __init__(self, all_params, model_spec, device, output_path, service_port):
        self.base_url = "http://localhost:" + str(service_port)
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.test_payloads_path = "utils/test_payloads"

    def get_health(self, attempt_number = 1) -> bool:
        import requests
        response = requests.get(f"{self.base_url}/tt-liveness")
        # server returns 200 if healthy only
        # otherwise it is 405
        if response.status_code != 200:
            if attempt_number < 20:
                print(f"Health check failed with status code: {response.status_code}. Retrying...")
                time_module.sleep(15)
                return self.get_health(attempt_number + 1)
            else:
                raise Exception(f"Health check failed with status code: {response.status_code}")
        return (True, response.json().get("runner_in_use", None))


    def _read_latest_benchmark_json(self) -> dict:
        """Read the latest benchmark JSON file based on timestamp."""
        import glob
        import json
        
        # Pattern to match benchmark JSON files
        # Remove 'evals_output' from the path if present
        output_path_clean = str(Path(self.output_path)).replace("evals_output", "").rstrip("/")
        # Look in the benchmarks_output subdirectory
        pattern = str(Path(output_path_clean) / "benchmarks_output" / f"benchmark_{self.model_spec.model_id}_*.json")
        
        # Find all matching files
        json_files = glob.glob(pattern)
        
        if not json_files:
            raise FileNotFoundError(f"No benchmark JSON files found matching pattern: {pattern}")
        
        # Sort by modification time to get the latest
        latest_file = max(json_files, key=lambda x: Path(x).stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)

    def run_evals(self) -> list[SDXLTestStatus]:
        import json
        # Read the latest benchmark JSON file and extract ttft
        benchmark_data = self._read_latest_benchmark_json()
        ttft = benchmark_data.get("benchmarks", {}).get("ttft", 0)
        
        # TODO: Compare ttft with other variables here
        print(f"Extracted TTFT value: {ttft}")
        
        benchmark_data['evals'] = {
            "model": self.model_spec.model_id,
            "task_name": self.all_params.tasks[0].task_name,
            "tolerance": self.all_params.tasks[0].score.tolerance,
            "published_score": self.all_params.tasks[0].score.published_score,
            "score": ttft,
            "publishsed_score_ref": self.all_params.tasks[0].score.published_score_ref,
        }
        
        # Write benchmark_data to JSON file
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"/ self.model_spec.hf_model_repo.replace('/', '__') / f"results_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        print(f"Evaluation data written to: {eval_filename}")
        

    def run_benchmarks(self, attempt = 0) -> list[SDXLTestStatus]:
        try:
            (health_status, runner_in_use) = self.get_health()
            if health_status:
                print("Health check passed.")
            else:
                print("Health check failed.")
                return False

            # Get num_calls from CNN benchmark parameters
            cnn_params = next((param for param in self.all_params if hasattr(param, 'num_eval_runs')), None)
            num_calls = cnn_params.num_eval_runs if cnn_params and hasattr(cnn_params, 'num_eval_runs') else 2

            status_list = []
            
            is_image_generate_model = runner_in_use.startswith("tt-sd")
            is_audio_transcription_model = "whisper" in runner_in_use
            
            if runner_in_use and is_image_generate_model:
                for i in range(1):
                    print(f"Generating image {i + 1}/{num_calls}...")
                    status, elapsed = self._generate_image()
                    inference_steps_per_second = 20 / elapsed if elapsed > 0 else 0
                    print(f"Generated image with {20} steps in {elapsed:.2f} seconds.")
                    status_list.append(SDXLTestStatus(
                        status=status,
                        elapsed=elapsed,
                        num_inference_steps=20,
                        inference_steps_per_second=inference_steps_per_second
                    ))
            elif runner_in_use and is_audio_transcription_model:
                for i in range(num_calls):
                    print(f"Transcribing audio {i + 1}/{num_calls}...")
                    status, elapsed = self._transcribe_audio()
                    print(f"Transcribed audio with {50} steps in {elapsed:.2f} seconds.")
                    status_list.append(SDXLTestStatus(
                        status=status,
                        elapsed=elapsed,
                    ))
            elif runner_in_use and not is_image_generate_model:
                for i in range(num_calls):
                    print(f"Analyizing image {i + 1}/{num_calls}...")
                    status, elapsed = self._analyze_image()
                    print(f"Generated image with {50} steps in {elapsed:.2f} seconds.")
                    status_list.append(SDXLTestStatus(
                        status=status,
                        elapsed=elapsed,
                    ))


            return self._generate_report(status_list, is_image_generate_model)
        except Exception as e:
            print(f"Health check encountered an error: {e}")
            return False

    def _generate_report(self, status_list: list[SDXLTestStatus], is_image_generate_model: bool) -> None:
        import json
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert SDXLTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                    "num_requests": len(status_list),
                    "num_inference_steps": status_list[0].num_inference_steps if status_list and is_image_generate_model else 0,
                    "ttft": sum(status.elapsed for status in status_list) / len(status_list) if status_list else 0,
                    "inference_steps_per_second": sum(status.inference_steps_per_second for status in status_list) / len(status_list) if status_list and is_image_generate_model else 0,
                },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime()),
            "task_type": "cnn"
        }
        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"Report generated: {result_filename}")
        return True

    def _generate_image(self, num_inference_steps: int = 20) -> tuple[bool, float]:
        import requests
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": "Rabbit",
            "seed": 0,
            "guidance_scale": 3.0,
            "number_of_images": 1,
            "num_inference_steps": num_inference_steps
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/image/generations", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        return (response.status_code == 200), elapsed
    
    def _analyze_image(self) -> tuple[bool, float]:
        import requests
        with open(f"{self.test_payloads_path}/image_client_image_payload.txt", "r") as f:
            imagePayload = f.read()
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": imagePayload
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/cnn/search-image", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        return (response.status_code == 200), elapsed
    
    def _transcribe_audio(self) -> tuple[bool, float]:
        import requests
        with open(f"{self.test_payloads_path}/image_client_audio_payload.txt", "r") as f:
            audioFile = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=headers, timeout=90)
        print(f"Transcribed audio: {response.json()}")
        elapsed = time() - start_time
        return (response.status_code == 200), elapsed
