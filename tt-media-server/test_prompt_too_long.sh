#!/bin/bash
# Test that server handles prompt exceeding max_model_len gracefully
curl -sv -H "Authorization: Bearer your-secret-key" -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Llama-3.2-3B-Instruct","prompt":"word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word","max_tokens":32,"stream":true}' \
  http://localhost:8000/v1/completions
echo
