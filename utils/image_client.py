# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from pathlib import Path
from time import time
import time as time_module
from typing import Optional
import base64


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
    def __init__(self, all_params=None, model_spec=None, device=None, output_path=None, service_port=None, base_url=None, jwt_secret=None, image_path=None):
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = "http://localhost:" + str(service_port)
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.jwt_secret = jwt_secret or "your-secret-key"
        self.image_path = image_path

    def _load_image_as_base64(self, image_path: Optional[str] = None) -> str:
        """Load an image from file path and convert to base64 data URI.
        
        Args:
            image_path: Path to the image file. If None, uses self.image_path
            
        Returns:
            Base64 data URI string in format: data:image/{type};base64,{data}
        """
        path_to_use = image_path or self.image_path
        if not path_to_use:
            raise ValueError("No image path provided")
        
        image_file = Path(path_to_use)
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_file}")
        
        # Determine image type from file extension
        file_ext = image_file.suffix.lower()
        mime_type_map = {
            '.jpg': 'jpeg', '.jpeg': 'jpeg',
            '.png': 'png', '.gif': 'gif',
            '.bmp': 'bmp', '.webp': 'webp'
        }
        mime_type = mime_type_map.get(file_ext, 'jpeg')
        
        # Read and encode image
        with open(image_file, 'rb') as f:
            image_data = f.read()
        
        encoded_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/{mime_type};base64,{encoded_data}"

    def get_health(self, attempt_number = 1) -> bool:
        import requests
        response = requests.get(f"{self.base_url}/image/tt-liveness")
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
            
            if runner_in_use and is_image_generate_model:
                for i in range(num_calls):
                    print(f"Generating image {i + 1}/{num_calls}...")
                    status, elapsed = self.generate_image()
                    inference_steps_per_second = 20 / elapsed if elapsed > 0 else 0
                    print(f"Generated image with {20} steps in {elapsed:.2f} seconds.")
                    status_list.append(SDXLTestStatus(
                        status=status,
                        elapsed=elapsed,
                        num_inference_steps=20,
                        inference_steps_per_second=inference_steps_per_second
                    ))
            elif runner_in_use and not is_image_generate_model:
                for i in range(num_calls):
                    print(f"Analyizing image {i + 1}/{num_calls}...")
                    status, elapsed = self.analyze_image()
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
        # Convert SDXLTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                    "num_requests": len(status_list),
                    "num_inference_steps": status_list[0].num_inference_steps if status_list and is_image_generate_model else 0,
                    "ttft": sum(status.elapsed for status in status_list) / len(status_list) if status_list else 0,
                    "inference_steps_per_second": sum(status.inference_steps_per_second for status in status_list) / len(status_list) if status_list and is_image_generate_model else 0,
                },
            "model": self.model_spec.model_id,
            "device": self.device.name,
            "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S", time_module.localtime()),
            "task_type": "cnn"
        }
        with open(result_filename, "w") as f:
            json.dump(report_data, f, indent=4)
        print(f"Report generated: {result_filename}")
        return True

    def generate_image(self, num_inference_steps: int = 20) -> tuple[bool, float]:
        import requests
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.jwt_secret}",
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": "Michael Jordan blocked by Spud Webb",
            "num_inference_step": num_inference_steps
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/image/generations", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        return (response.status_code == 200), elapsed
    
    def analyze_image(self, image_path: Optional[str] = None) -> tuple[bool, float]:
        import requests
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.jwt_secret}",
            "Content-Type": "application/json"
        }
        
        try:
            image_base64 = self._load_image_as_base64(image_path)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error loading image: {e}")
            return False, 0.0
            
        payload = {
            "prompt": image_base64
        }
        start_time = time()
        response = requests.post(f"{self.base_url}/cnn/search-image", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        return (response.status_code == 200), elapsed
    
    def search_image(self, image_data: str):
        """Search/analyze image using CNN inference endpoint.
        
        Args:
            image_data: Either a base64 encoded image data URI or a file path to an image
            
        Returns:
            requests.Response: The HTTP response from the CNN inference endpoint
        """
        import requests
        
        # Check if input is a file path or base64 data
        if image_data.startswith('data:image/'):
            # Already base64 data URI
            image_base64 = image_data
        else:
            # Assume it's a file path
            try:
                image_base64 = self._load_image_as_base64(image_data)
            except (ValueError, FileNotFoundError) as e:
                raise ValueError(f"Invalid image data or file path: {e}")
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.jwt_secret}",
            "Content-Type": "application/json"
        }
        payload = {
            "image": image_base64
        }
        return requests.post(f"{self.base_url}/cnn/search-image", json=payload, headers=headers, timeout=90)