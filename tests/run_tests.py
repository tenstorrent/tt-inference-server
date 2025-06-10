import argparse
import os
import json
import jwt
import sys
# Add the script's directory to the Python path
# this for 0 setup python setup script
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent

# Add the project root to the Python path to ensure imports work
sys.path.insert(0, str(project_root))

from tests import Tests
from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import DeviceTypes
from workflows.workflow_config import (
    WORKFLOW_TESTS_CONFIG,
)
from workflows.log_setup import setup_workflow_script_logger
import logging
from workflows.utils import get_model_id
logger = logging.getLogger(__name__)

def parse_arguments():
    valid_impls = {config.impl.impl_name for _, config in MODEL_CONFIGS.items()}
    parser = argparse.ArgumentParser(description="Run Tests.")
    parser.add_argument("--run-mode", type=str, 
                       choices=["single", "multiple", "validated"],
                       help="Run mode: single (explicit params), multiple (generated matrix), or validated (model config combinations)", 
                       default=argparse.SUPPRESS)
    parser.add_argument("--endurance-mode", action="store_true", help="Runs continuously for 24 hours", default=argparse.SUPPRESS)
    parser.add_argument("--max-context-length", type=int, help="Useful for CLI single-run prompting", default=argparse.SUPPRESS)
    parser.add_argument("--input-size", type=int, help="Input token length", default=argparse.SUPPRESS)
    parser.add_argument("--output-size", type=int, help="Output token length", default=argparse.SUPPRESS)
    parser.add_argument("--max-concurrent", type=int, help="Optional max_concurrent (Like-Batch Size) (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--num-prompts", type=int, help="num_prompts, (Like # of Users) (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--service-port", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, default=argparse.SUPPRESS)
    parser.add_argument('--device', type=str, help='The device to use: N150, N300, T3K, TG')
    parser.add_argument('--project-root', type=Path, default=project_root)
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--disable-trace-capture",
        action="store_true",
        help="Disables trace capture requests, use to speed up execution if inference server already running and traces captured.",
    )
    parser.add_argument(
        "--impl",
        required=False,
        choices=valid_impls,
        help=f"Implementation option (choices: {', '.join(valid_impls)})",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID",
        default="",
    )

    return parser.parse_args()

if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")
    args = parse_arguments()

    if args.jwt_secret:
        # If jwt-secret is provided, generate the JWT and set OPENAI_API_KEY.
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, args.jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    
    # Get the model ID based on impl and model name
    model_id = get_model_id(args.impl, args.model, args.device)
    
    # Check if the model ID exists in MODEL_CONFIGS
    if model_id not in MODEL_CONFIGS:
        raise ValueError(
            f"No model configuration found for model_id: {model_id} (impl: {args.impl}, model: {args.model})"
        )
    
    # Get the model configuration
    model_config = MODEL_CONFIGS[model_id]
    
    # Convert device string to DeviceTypes enum
    device = DeviceTypes.from_string(args.device)
    
    # Check if the device matches the model configuration's device type
    if device != model_config.device_type:
        raise ValueError(
            f"Device {args.device} does not match the model configuration device type {model_config.device_type.name} for model: {model_config.model_name}"
        )
    
    workflow_config = WORKFLOW_TESTS_CONFIG
    logger.info(f"workflow_config=: {workflow_config}")
    logger.info(f"model_config=: {model_config}")
    logger.info(f"device=: {args.device}")
    logger.info(f"service_port=: {args.service_port}")
    if hasattr(args, "output_path"):
        logger.info(f"output_path=: {args.output_path}")
    else:
        args.output_path = str(project_root) + "/workflow_logs/tests_output"
        logger.info(f"output_path=: {args.output_path}")
    logger.info("Wait for the vLLM server to be ready ...")

    run_test = Tests(args, model_config)

    run_test.run()
    logger.info("✅ Completed tests")
