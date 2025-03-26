import argparse

# Add the script's directory to the Python path
# this for 0 setup python setup script
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

from tests import Tests
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.workflow_config import (
    WORKFLOW_TESTS_CONFIG,
)
from workflows.log_setup import setup_workflow_script_logger
import logging

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Tests.")
    parser.add_argument("--server_start", action="store_true", help="Start the server and execute the command.")
    parser.add_argument("--mode", type=str, default="max_seq", help="Test mode: max_seq or continuous_batch")
    parser.add_argument("--run_mode", type=str, help="Run mode: single or multiple", default=argparse.SUPPRESS)
    parser.add_argument("--max_context_length", type=int, help="Useful for CLI single-run prompting", default=argparse.SUPPRESS)
    parser.add_argument("--input_size", type=str, help="Input token length", default=argparse.SUPPRESS)
    parser.add_argument("--output_size", type=str, help="Output token length", default=argparse.SUPPRESS)
    parser.add_argument("--max_concurrent", type=str, help="Optional max_concurrent (Like-Batch Size) (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--num_prompts", type=str, help="num_prompts, (Like # of Users) (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--local_env_file", type=str, help="Local Environment File.", default=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--jwt-secret", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--service-port", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default=argparse.SUPPRESS)
    parser.add_argument('--device', type=str, help='The device to use: N150, N300, T3K, TG')
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already running and traces captured.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")
    args = parse_arguments()
    # if args.jwt_secret:
    import os
    import json
    import jwt
    if "JWT_SECRET" not in os.environ: #TODO better handling of JWT_SECRET
        os.environ["JWT_SECRET"] = args.jwt_secret
    json_payload = json.loads(
        '{"team_id": "tenstorrent", "token_id": "debug-test"}'
    )
    encoded_jwt = jwt.encode(json_payload, os.environ["JWT_SECRET"], algorithm="HS256")
    os.environ["OPENAI_API_KEY"] = encoded_jwt
    logger.info(
        "OPENAI_API_KEY environment variable set using provided JWT secret."
    )

    if args.model not in MODEL_CONFIGS:
        raise ValueError(
            f"No evaluation tasks defined for model: {args.model}"
        )
    model_config = MODEL_CONFIGS[args.model]
    device = DeviceTypes.from_string(args.device)
    workflow_config = WORKFLOW_TESTS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    logger.info(f"service_port=: {args.service_port}")
    logger.info(f"output_path=: {args.output_path}")

    logger.info("Wait for the vLLM server to be ready ...")

    if hasattr(args, "max_context_length"): #TODO: run_mode should override this #TODO Test candidate
        print("Using user input max_context_length")
        run_test = Tests(args)
    else:
        run_test = Tests(args)

    run_test.run()
    logger.info("✅ Completed tests")
