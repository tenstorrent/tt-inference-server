from pathlib import Path
from time import time
import time as time_module


class SDXLTestStatus:
    status: bool
    elapsed: float
    num_inference_steps: int
    inference_steps_per_second: float

    def __init__(self, status: bool, elapsed: float, num_inference_steps: int, inference_steps_per_second: float):
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
        return True

    def generate_image(self, num_inference_steps: int = 20) -> tuple[bool, float]:
        import requests
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
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

    def run_benchmarks(self, attempt = 0) -> list[SDXLTestStatus]:
        try:
            health_status = self.get_health()
            if health_status:
                print("Health check passed.")
            else:
                print("Health check failed.")
                return False

            num_calls = 2

            status_list = []
            
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


            return self._generate_report(status_list)
        except Exception as e:
            print(f"Health check encountered an error: {e}")
            return False
    
    def _generate_report(self, status_list: list[SDXLTestStatus]) -> None:
        import json
        result_filename = (
            Path(self.output_path)
            / f"benchmark_{self.model_spec.model_id}_{time()}.json"
        )
        # Convert SDXLTestStatus objects to dictionaries for JSON serialization
        report_data = {
            "benchmarks": {
                    "num_requests": len(status_list),
                    "num_inference_steps": status_list[0].num_inference_steps if status_list else 0,
                    "ttft": sum(status.elapsed for status in status_list) / len(status_list) if status_list else 0,
                    "inference_steps_per_second": sum(status.inference_steps_per_second for status in status_list) / len(status_list) if status_list else 0,
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