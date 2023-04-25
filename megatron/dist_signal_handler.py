import signal

import torch


def get_world_size():
    return (
        torch.distributed.get_world_size()
        if torch.distributed.is_available()
        and torch.distributed.is_initialized()
        else 1
    )


def get_device(local_rank=None):
    backend = torch.distributed.get_backend()
    if backend == 'nccl':
        if local_rank is None:
            device = torch.device('cuda')
        else:
            device = torch.device(f'cuda:{local_rank}')
    elif backend == 'gloo':
        device = torch.device('cpu')
    else:
        raise RuntimeError
    return device


def all_gather_item(item, dtype, group=None, async_op=False, local_rank=None):
    if not torch.distributed.is_available() or \
       not torch.distributed.is_initialized():
        return [item]

    device = get_device(local_rank)

    group_size = group.size() if group is not None else get_world_size()
    tensor = torch.tensor([item], device=device, dtype=dtype)
    output_tensors = [
        torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        for _ in range(group_size)
    ]
    torch.distributed.all_gather(output_tensors, tensor, group, async_op)
    return [elem.item() for elem in output_tensors]


class DistributedSignalHandler:
    def __init__(self, sig=signal.SIGTERM):
        self.sig = sig

    def signals_received(self):
        return all_gather_item(self._signal_received, dtype=torch.int32)

    def __enter__(self):
        self._signal_received = False
        self.released = False
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self._signal_received = True

        signal.signal(self.sig, handler)

        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self.released:
            return False

        signal.signal(self.sig, self.original_handler)
        self.released = True
        return True
