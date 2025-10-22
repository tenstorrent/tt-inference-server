# TT-Inference-Server Demo Guide

## ✅ Current Status

Your tt-inference-server is **UP AND RUNNING**! 

- **Model**: Llama-3.2-1B-Instruct
- **Device**: n150
- **Port**: 8000
- **Container ID**: e7b66381a403
- **Status**: Server is ready and responding to requests

## 🚀 Quick Test Commands

### 1. Use the Test Script (Recommended)

```bash
cd /home/ttuser/aperezvicente/tt-inference-server

# Run with custom prompt
python3 test_server.py "Your question here"

# Run demo suite (multiple examples)
python3 test_server.py
```

### 2. Use curl (Manual)

First, generate JWT token:
```python
python3 << 'EOF'
import jwt
payload = {"team_id": "tenstorrent", "token_id": "debug-test"}
token = jwt.encode(payload, "tenstorrent", algorithm="HS256")
print(token)
EOF
```

Then use the token with curl:
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <YOUR_TOKEN_HERE>" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }' | jq .
```

## 📊 Server Management

### Check Server Status
```bash
# List running containers
docker ps

# View real-time logs
docker logs -f e7b66381a403

# View last 50 lines
docker logs --tail 50 e7b66381a403
```
workflows/run_local_server.py
### Stop the Server
```bash
docker stop e7b66381a403
```

### Restart the Server
```bash
cd /home/ttuser/aperezvicente/tt-inference-server
python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow server --docker-server --skip-system-sw-validation
```

## 🎯 Example Prompts to Try

### Simple Q&A
```bash
python3 test_server.py "What is the capital of Japan?"
```

### Code Generation
```bash
python3 test_server.py "Write a Python function to sort a list:"
```

### Explanation
```bash
python3 test_server.py "Explain quantum computing in simple terms:"
```

### Math
```bash
python3 test_server.py "Solve: 2x + 5 = 15"
```

## 📈 Next Steps

### 1. Run Benchmarks
Measure performance of your model:
```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow benchmarks \
  --docker-server \
  --skip-system-sw-validation
```

Results will be in: `workflow_logs/benchmarks_output/`

### 2. Run Evaluations
Test model accuracy:
```bash
python3 run.py \
  --model Llama-3.2-1B-Instruct \
  --device n150 \
  --workflow evals \
  --docker-server \
  --skip-system-sw-validation
```

Results will be in: `workflow_logs/evals_output/`

### 3. Try Different Models
```bash
# Larger model (needs more resources)
python3 run.py \
  --model Llama-3.2-3B-Instruct \
  --device n300 \
  --workflow server \
  --docker-server \
  --skip-system-sw-validation

# 8B model
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n300 \
  --workflow server \
  --docker-server \
  --skip-system-sw-validation
```

### 4. Connect Open-WebUI
You have Open-WebUI running on port 3000. Configure it to connect to:
- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: Generate JWT token (see above)

Access Open-WebUI at: http://localhost:3000

### 5. Load Testing
Use the locust tool for load testing:
```bash
cd /home/ttuser/aperezvicente/tt-inference-server/locust
# Follow instructions in locust/README.md
```

## 📝 Configuration Files

- **Environment Variables**: `.env`
- **Model Weights**: `persistent_volume/`
- **Logs**: `workflow_logs/`
- **Run Specs**: `workflow_logs/run_specs/`

## 🔧 Troubleshooting

### Server Not Responding
```bash
# Check if container is running
docker ps

# Check logs for errors
docker logs e7b66381a403 | tail -100

# Restart if needed
docker restart e7b66381a403
```

### Authentication Errors
The server requires JWT authentication. Make sure you:
1. Use the correct JWT_SECRET (currently: "tenstorrent")
2. Generate token with correct payload: `{"team_id": "tenstorrent", "token_id": "debug-test"}`
3. Include header: `Authorization: Bearer <token>`

### Out of Memory
If you see OOM errors, try:
- Using a smaller model (1B instead of 3B)
- Reducing batch size
- Checking available device memory

## 📚 Documentation

- Main README: [README.md](README.md)
- Quick Start: [QUICK_START.md](QUICK_START.md)
- Workflows Guide: [docs/workflows_user_guide.md](docs/workflows_user_guide.md)
- Development: [docs/development.md](docs/development.md)

## 🎉 Success!

Your tt-inference-server is successfully running and serving requests!

Generated response example:
```
Prompt: "What is the capital of France? Answer:"
Response: "Paris"
Tokens: 10 prompt + 32 completion = 42 total
```

Happy inferencing! 🚀

