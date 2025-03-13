import argparse
from tests import Tests

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Tests.")
    parser.add_argument("--server_start", action="store_true", help="Start the server and execute the command.")
    parser.add_argument("--mode", type=str, default="max_seq", help="Test mode: max_seq or continuous_batch")
    parser.add_argument("--run_mode", type=str, default="single", help="Run mode: single or multiple")
    parser.add_argument("--max_context_length", type=str, help="placeholder; possibly redundant", default=argparse.SUPPRESS)
    parser.add_argument("--input_size", type=str, help="Input token length", default=argparse.SUPPRESS)
    parser.add_argument("--output_size", type=str, help="Output token length", default=argparse.SUPPRESS)
    parser.add_argument("--batch_size", type=str, help="Optional Batch Size AKA max_concurrent (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--users", type=str, help="Optional number of Users AKA num_prompts (default: 1).", default=argparse.SUPPRESS)
    parser.add_argument("--local_env_file", type=str, help="Local Environment File.", default=argparse.SUPPRESS)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_test = Tests(args, server_start=args.server_start)
    run_test.run()
