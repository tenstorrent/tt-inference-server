import os
import traceback

from domain.completion_request import CompletionRequest
from domain.completion_response import CompletionStreamChunk
from telemetry.telemetry_client import TelemetryEvent
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.text_utils import TextUtils
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams


class VLLMForgeRunner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.llm_engine: AsyncLLMEngine | None = None

    def set_device(self):
        return {}

    @log_execution_time(
        "VLLM Forge model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def load_model(self) -> bool:
        self.logger.info(f"Device {self.device_id}: Loading VLLM Forge model...")

        engine_args = AsyncEngineArgs(
            model="meta-llama/Llama-3.1-8B-Instruct",
            max_model_len=65536,
            max_num_seqs=8,
            enable_chunked_prefill=False,
            block_size=64,
            max_num_batched_tokens=65536,
            scheduler_delay_factor=0.0,  # Try 0.0 (no delay) first
            seed=9472,
        )

        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Warmup
        prompt = "Hello, it's me"
        warmup_params = SamplingParams(temperature=0.0, max_tokens=10)

        self.logger.info(f"Device {self.device_id}: Starting model warmup")
        async for _ in self.llm_engine.generate(prompt, warmup_params, "warmup"):
            pass

        self.logger.info(f"Device {self.device_id}: Model warmup completed")
        return True

    @log_execution_time(
        "Run VLLM Forge inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def run_inference(self, requests: list[CompletionRequest]):
        if self.llm_engine is None:
            raise RuntimeError("Model not loaded")

        request = requests[0]

        sp = SamplingParams(
            temperature=request.temperature or 0.8,
            top_p=request.top_p or 0.95,
            max_tokens=request.max_tokens or 16,
        )

        return self._streaming_generator(request, sp)

    async def _streaming_generator(self, request: CompletionRequest, sampling_params):
        self.logger.debug(f"[{request._task_id}] start streaming")
        try:
            # **FIX**: add_request() is sync, returns AsyncStream directly
            # No await needed!
            stream = await self.llm_engine.add_request(
                request_id=request._task_id,
                prompt=request.prompt,
                params=sampling_params,
            )

            previous_text = ""

            # **KEY**: Use while loop with await stream.get()
            # This doesn't block other coroutines
            while True:
                # **NON-BLOCKING**: await yields control to event loop
                request_output = await stream.get()

                if request_output.finished:
                    yield {
                        "type": "final_result",
                        "result": CompletionStreamChunk(text=previous_text),
                        "task_id": request._task_id,
                        "return": False,
                    }
                    break  # Exit loop when done

                for output in request_output.outputs:
                    current_text = output.text
                    delta_text = current_text[len(previous_text) :]
                    previous_text = current_text

                    cleaned_delta = TextUtils.clean_text(delta_text)

                    if not cleaned_delta:
                        continue

                    yield {
                        "type": "streaming_chunk",
                        "chunk": CompletionStreamChunk(text=cleaned_delta),
                        "task_id": request._task_id,
                    }

            self.logger.debug(f"[{request._task_id}] streaming complete")

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Error for {request._task_id}: {e}\n{traceback.format_exc()}"
            )
            raise

    async def _non_streaming_generation(self, request: CompletionRequest, params):
        """
        Non-streaming returns a list, not a generator
        """
        self.logger.debug(f"[{request._task_id}] non-streaming start")

        result_text = ""

        async for req_out in self.llm_engine.generate(
            request.prompt, params, request._task_id
        ):
            if req_out.outputs:
                result_text = TextUtils.clean_text(req_out.outputs[0].text)
                break

        self.logger.debug(f"[{request._task_id}] non-streaming complete")
        return [CompletionStreamChunk(text=result_text)]
