import pathlib
from datetime import datetime

from inference_config import inference_config

workers = 1
# use 0.0.0.0 for externally accessible
bind = f"0.0.0.0:{inference_config.backend_server_port}"
reload = False
worker_class = "gthread"
threads = 96
timeout = 120

# set log files
if not pathlib.Path(inference_config.log_cache).exists():
    pathlib.Path(inference_config.log_cache).mkdir(parents=True, exist_ok=True)
datetime_prefix = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")
accesslog = f"{inference_config.log_cache}/{datetime_prefix}_access.log"
errorlog = f"{inference_config.log_cache}/{datetime_prefix}_error.log"
loglevel = "info"

# switch between mock model and production version
if inference_config.mock_model:
    wsgi_app = "_mock_inference_api_server:create_test_server()"
else:
    wsgi_app = "inference_api_server:create_server()"
