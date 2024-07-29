# TT Metalium Mistral 7B Inference API

Model: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

#### Using docker compose

To run the service foregrounded for python interactive debugging with pdf breakpoints use the docker-compose.yml command: `command: ["/bin/bash", "-c", "sleep infinity"]`

```bash
docker compose up

export IMAGE_NAME='project-falcon/tt-metal-mistral7b:v0.0.1'
docker exec -it $(docker ps | grep "${IMAGE_NAME}" | awk '{print $1}') bash

cd /mnt/src
gunicorn --config gunicorn.conf.py
```

### JWT_TOKEN Authorization

To authenticate requests use the header `Authorization`. The JWT token can be computed using the script `jwt_util.py`. This is an example:
```bash
export JWT_ENCODED=$(/mnt/scripts/jwt_util.py --secret ${JWT_SECRET} encode '{"team_id": "tenstorrent", "token_id":"debug-test"}')
export JWT_TOKEN="Bearer ${JWT_ENCODED}"
```