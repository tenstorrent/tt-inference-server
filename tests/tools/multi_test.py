# multi_test.py
from datetime import datetime
from .single_test import single_benchmark_execution

def mass_benchmark_execution(args):
    log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    single_benchmark_execution(args, log_timestamp)
    return

