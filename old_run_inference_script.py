    @log_execution_time("Lora Inference")
    def run(self, request: CompletionRequest):
        max_cache_len: int = 128
        batch_size: int = 1

        requested_mode = "base" if request.use_base_model else "fine_tuned"

        if requested_mode != self._compiled_mode:
            if self.compiled_inference_model is not None:
                del self.compiled_inference_model
                self._active_model.to("cpu")
                torch._dynamo.reset()

            if request.use_base_model:
                self._active_model = self.hf_base_model_inference
                self.logger.info("Using base model for inference")
            else:
                self._active_model = self.hf_fine_tuned_model_inference
                self.logger.info(f"Using fine-tuned model for inference from {PEFT_MODEL_PATH}")
            
            self._active_model.to(self.device)
            self.compiled_inference_model = torch.compile(
                self._active_model, backend="tt"
            )
            self.compiled_inference_model.eval()
            self._compiled_mode = requested_mode

        user_prompt = [request.prompt]

        input_args = self._construct_inputs(
            user_prompt, 
            self._active_model.config, batch_size, max_cache_len,
        )

        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        input_args = self._transfer_inputs_to_device(input_args, self.device)

        output_tokens = self._run_generate(
            self.compiled_inference_model,
            input_args,
            self.device,
            max_tokens_to_generate,
            user_prompt,
        )

        return ["".join(output_tokens)]