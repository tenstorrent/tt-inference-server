# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from time import time
import time as time_module
from typing import Optional


class AudioTestStatus:
    status: bool
    elapsed: float
    ttft: Optional[float]

    def __init__(self, status: bool, elapsed: float, ttft: float = 0):
        self.status = status
        self.elapsed = elapsed
        self.ttft = ttft


class AudioClient:
    def __init__(self, all_params, model_spec, device, output_path, service_port):
        self.base_url = "http://localhost:" + str(service_port)
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.test_payloads_path = "utils/test_payloads"

    def get_health(self, attempt_number=1) -> bool:
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
        import json
        import glob
        
        # Find all benchmark JSON files for this model
        pattern = f"{self.output_path}/benchmark_{self.model_spec.model_id}_*.json"
        benchmark_files = glob.glob(pattern)
        
        if not benchmark_files:
            raise FileNotFoundError(f"No benchmark files found matching pattern: {pattern}")
        
        # Sort by modification time and get the latest
        latest_file = max(benchmark_files, key=lambda f: Path(f).stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            return json.load(f)

    def run_evals(self):
        """Convert benchmark data to evaluation format."""
        import json
        
        # Read the latest benchmark data
        benchmark_data = self._read_latest_benchmark_json()
        
        # Create evaluation filename
        eval_filename = (
            Path(self.output_path)
            / f"eval_{self.model_spec.model_id}"
            / f"{self.model_spec.hf_model_repo.replace('/', '__')}"
            / f"results_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        eval_filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(eval_filename, "w") as f:
            json.dump(benchmark_data, f, indent=4)
        print(f"Evaluation data written to: {eval_filename}")

    def run_benchmarks(self, attempt=0) -> list[AudioTestStatus]:
        try:
            (health_status, runner_in_use) = self.get_health()
            if health_status:
                print("Health check passed.")
            else:
                print("Health check failed.")
                return False

            # Get num_calls from audio benchmark parameters
            audio_params = next((param for param in self.all_params if hasattr(param, 'num_eval_runs')), None)
            num_calls = audio_params.num_eval_runs if audio_params and hasattr(audio_params, 'num_eval_runs') else 2

            status_list = []
            
            is_audio_transcription_model = "whisper" in runner_in_use
            
            if runner_in_use and is_audio_transcription_model:
                import asyncio
                for i in range(num_calls):
                    print(f"Transcribing audio {i + 1}/{num_calls}...")
                    status, elapsed, ttft = asyncio.run(self._transcribe_audio())
                    print(f"Transcribed audio in {elapsed:.2f} seconds.")
                    status_list.append(AudioTestStatus(
                        status=status,
                        elapsed=elapsed,
                        ttft=ttft,
                    ))

            return self._generate_report(status_list)
        except Exception as e:
            print(f"Health check encountered an error: {e}")
            return False

    def _generate_report(self, status_list: list[AudioTestStatus]) -> None:
        import json
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time()}.json"
        )
        # Create directory structure if it doesn't exist
        result_filename.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert AudioTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                "num_requests": len(status_list),
                "ttft": sum(status.elapsed for status in status_list) / len(status_list) if status_list else 0,
            },
            "model": self.model_spec.model_name,
            "device": self.device.name,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime()),
            "task_type": "audio"
        }
        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"Report generated: {result_filename}")
        return True

    async def _transcribe_audio(self) -> tuple[bool, float, float]:
        """Transcribe audio without streaming - measures total latency
        Returns:
            (success, latency_sec, ttft_sec)
        """
        import requests
        
        # Read audio file
        with open(f"{self.test_payloads_path}/image_client_audio_payload.txt", "r") as f:
            audioFile = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile,
            "stream": False
        }
        
        start_time = time()
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        ttft = elapsed
        
        return (response.status_code == 200), elapsed, ttft
