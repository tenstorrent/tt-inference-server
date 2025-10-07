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
    ttft: Optional[float]

    def __init__(self, status: bool, elapsed: float, num_inference_steps: int = 0, inference_steps_per_second: float = 0, ttft: float = 0):
        self.status = status
        self.elapsed = elapsed
        self.num_inference_steps = num_inference_steps
        self.inference_steps_per_second = inference_steps_per_second
        self.ttft = ttft

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
                    status, elapsed, ttft = self._transcribe_audio()
                    print(f"Transcribed audio in {elapsed:.2f} seconds.")
                    status_list.append(SDXLTestStatus(
                        status=status,
                        elapsed=elapsed,
                        ttft=ttft,
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
            "task_type": "cnn" if is_image_generate_model else "audio"
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
    
    def _transcribe_audio(self) -> tuple[bool, float, float]:
        # Get streaming setting from model spec CLI args (default to True if not set)
        streaming_enabled = self.model_spec.cli_args.get('streaming', 'true').lower() == 'true'
        if streaming_enabled:
            return self._transcribe_audio_streaming_on()

        return self._transcribe_audio_streaming_off()

    
    def _transcribe_audio_streaming_off(self) -> tuple[bool, float, float]:
        """Transcribe audio without streaming - direct transcription of the entire audio file"""
        import requests
        with open(f"{self.test_payloads_path}/image_client_audio_payload.txt", "r") as f:
            audioFile = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile,
            "stream": False,
            "return_perf_metrics": True
        }
        
        start_time = time()
        response = requests.post(f"{self.base_url}/audio/transcriptions", json=payload, headers=headers, timeout=90)
        elapsed = time() - start_time
        ttft = elapsed
        
        return (response.status_code == 200), elapsed, ttft
    
    def _transcribe_audio_streaming_on(self) -> tuple[bool, float, float]:
        """Transcribe audio with streaming enabled - receives partial results
        Measures:
            - TTFT (time to first token)
            - Total latency (end-to-end)
            - Tokens per second (throughput)
        Returns:
            (success, latency_sec, ttft_sec)
        """
        import requests
        import time
        import json
        import threading
        import queue
        import collections
        import re
        
        with open(f"{self.test_payloads_path}/image_client_audio_payload.txt", "r") as f:
            audioFile = f.read()

        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer your-secret-key",
            "Content-Type": "application/json"
        }
        payload = {
            "file": audioFile,
            "stream": True,
            "return_perf_metrics": True
        }
        
        q = queue.Queue()
        done = threading.Event()
        
        url = f"{self.base_url}/audio/transcriptions"
        metrics_log = []               # list of dicts: per-chunk metrics
        total_tokens = 0
        ttft = None
        rolling = collections.deque()  # stores tuples (recv_time, token_count)
        seen_ids = set()               # track seen chunk_ids to handle duplicates
        dedupe_chunk_ids = False       # set to True if you want to dedupe by chunk_id
        window_sec = 1.0               # rolling window for TPS calculation
        start_time = time.time()
        
        def producer():
            try:
                with requests.post(url, json=payload, headers=headers, stream=True, timeout=90) as resp:
                    if resp.status_code != 200:
                        q.put(("error", resp.status_code))
                        done.set()
                        return
                    for raw_line in resp.iter_lines(decode_unicode=True):
                        if not raw_line:
                            continue
                        # Put raw chunk into queue ASAP (consumer timestamps it)
                        q.put(("data", raw_line))
            except Exception as e:
                q.put(("error", str(e)))
            finally:
                done.set()
                
        # start background producer
        t = threading.Thread(target=producer, daemon=True)
        t.start()

        # Consumer: process immediately when data arrives
        last_chunk_time = time.time()
        max_idle_timeout = 10.0  # seconds - if no chunks received for this long, assume completion
        
        while not done.is_set() or not q.empty():
            # Calculate dynamic timeout: either until idle timeout or 1 second max
            time_since_last = time.time() - last_chunk_time
            remaining_idle = max_idle_timeout - time_since_last
            timeout = min(1.0, max(0.1, remaining_idle)) if remaining_idle > 0 else 0.1
            
            try:
                msg_type, payload_raw = q.get(timeout=timeout)
                last_chunk_time = time.time()  # Reset timeout on receiving data
            except queue.Empty:
                # Check if we've been idle too long
                if time.time() - last_chunk_time > max_idle_timeout:
                    print(f"[stream] Timeout: No chunks received for {max_idle_timeout}s, assuming completion")
                    break
                continue

            recv_time = time.time()  # <-- timestamp at reception (important)
            if msg_type == "error":
                # handle/return error
                print(f"[stream error] {payload_raw}")
                # attach metrics log to self for debugging / persistence
                self.streaming_metrics_log = metrics_log
                return False, 0.0, 0.0

            # parse chunk JSON
            try:
                chunk_obj = json.loads(payload_raw)
            except json.JSONDecodeError:
                # non-json chunk — skip but you might want to log
                continue

            text = chunk_obj.get("text", "")
            chunk_id = chunk_obj.get("chunk_id", None)
            
            # Check if this is the final summary chunk (contains segments, duration, etc.)
            # This indicates the transcription is complete
            if "segments" in chunk_obj and "duration" in chunk_obj and "task" in chunk_obj:
                print(f"[stream] Received final transcription summary with {len(chunk_obj.get('segments', []))} segments")
                # This is the final summary chunk, process it but then break
                break

            # optional dedupe by chunk_id (if your stream occasionally re-sends same id)
            if dedupe_chunk_ids and chunk_id is not None:
                if chunk_id in seen_ids:
                    # skip duplicate
                    continue
                seen_ids.add(chunk_id)

            # By your example, speaker markers appear as "[SPEAKER_01]" — treat them as non-tokens
            if re.match(r'^\[SPEAKER_\d+\]$', text.strip()):
                token_count = 0
            else:
                # simple tokenization: whitespace split. Replace with model tokenizer for exact token counts.
                token_count = len(text.strip().split()) if text.strip() else 0

            # TTFT: first **real** token arrival
            if ttft is None and token_count > 0:
                ttft = recv_time - start_time

            total_tokens += token_count

            # add to rolling window deque and evict old entries
            rolling.append((recv_time, token_count))
            cutoff = recv_time - window_sec
            while rolling and rolling[0][0] < cutoff:
                rolling.popleft()

            window_tokens = sum(c for _, c in rolling)
            rolling_tps = (window_tokens / window_sec) if window_sec > 0 else 0.0
            overall_tps = total_tokens / (recv_time - start_time) if (recv_time - start_time) > 0 else 0.0

            # Build per-chunk metric record
            record = {
                "recv_offset_sec": recv_time - start_time,
                "chunk_id": chunk_id,
                "text": text,
                "chunk_tokens": token_count,
                "cum_tokens": total_tokens,
                "rolling_tps": rolling_tps,
                "overall_tps": overall_tps
            }
            metrics_log.append(record)

            # Immediate feedback (you can change to structured logging or send to DB)
            print(f"[{record['recv_offset_sec']:.3f}s] id={chunk_id} tokens={token_count} "
                f"cum={total_tokens} window_tps={rolling_tps:.2f} overall_tps={overall_tps:.2f} text={text!r}")

        end_time = time.time()
        total_time = end_time - start_time

        # attach metrics to self for later inspection
        self.streaming_metrics_log = metrics_log

        # Safety: if no real token arrived, set ttft = total_time
        if ttft is None:
            ttft = total_time

        overall_tps_final = total_tokens / total_time if total_time > 0 else 0.0
        return True, total_time, ttft
        
        # return (response.status_code == 200), elapsed, ttft
        
        # Notes & suggestions

        # Tokenization accuracy: len(text.split()) is a quick approximation. If you want model token counts (for token-based cost or exact throughput), use the same tokenizer the model uses (e.g., tiktoken or sentencepiece) to compute tokens instead of whitespace split.
        # TTFT semantics: I only set TTFT on arrival of the first non-speaker token (i.e., token_count > 0). If you want TTFT to be the first any chunk (even speaker markers), change that condition.
        # Duplicate chunk IDs: In your sample chunk_id 1 appears twice. If chunk IDs can re-use numbers (per speaker or per segment), don’t dedupe by default. Enable dedupe_chunk_ids=True only if you know the server sometimes re-sends the exact same chunk and you want to ignore duplicates.
        # Rolling window: window_sec default is 1s. You can set it to 2s, 5s, etc., to smooth TPS.
        # Logging: self.streaming_metrics_log keeps everything; easily dump it to JSON/CSV for plotting later.