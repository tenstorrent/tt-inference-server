import torch


def set_torch_thread_limits(limits: int = 1):
    if torch.get_num_threads() != limits:
        torch.set_num_threads(limits)
    if torch.get_num_interop_threads() != limits:
        torch.set_num_interop_threads(limits)
