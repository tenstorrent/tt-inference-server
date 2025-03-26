import argparse
from tests import Tests

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
    parser.add_argument("--output-path", type=str, help="Not Implementated.", default=argparse.SUPPRESS)
    parser.add_argument("--log-path", type=str, help="Not Implementated.", default=argparse.SUPPRESS)
    parser.add_argument("--service-port", type=str, help="Not Implementated.", default=argparse.SUPPRESS)
    parser.add_argument("--model", type=str, help="Not Implementated.", default=argparse.SUPPRESS)
    parser.add_argument('--device', type=str, help='The device to use: N150, N300, T3K, TG')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    if hasattr(args, "max_context_length"):
        print("Using user input max_context_length")
        run_test = Tests(args)
    else:
        run_test = Tests(args)

    run_test.run()
