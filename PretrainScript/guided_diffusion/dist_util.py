"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
from mpi4py import MPI
import torch as th
import torch.distributed as dist

def setup_dist():
    """
    Setup a distributed process group for use with torchrun.
    """
    if dist.is_initialized():
        return
    
    # torchrun 已经设置了必要的环境变量
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # 选择合适的后端
    backend = "gloo" if not th.cuda.is_available() else "nccl"
    print(f'Using backend {backend}, rank={rank}, local_rank={local_rank}, world_size={world_size}')
    
    # 将当前进程绑定到相应的 GPU
    if th.cuda.is_available():
        th.cuda.set_device(local_rank)
        print(f'Process {rank} using GPU {local_rank}')
    
    # 初始化进程组
    dist.init_process_group(backend=backend)

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")

def load_state_dict(path, **kwargs):
    # 设定设备位置
    map_location = kwargs.pop("map_location", lambda storage, loc: storage.cuda(dev().index))
    
    # 仅打印一次
    if dist.get_rank() == 0:
        print(f'Loading state dict from {path}')
    
    # 直接在每个进程上加载
    state_dict = th.load(path, map_location=map_location, **kwargs)
    
    if dist.get_rank() == 0:
        print(f'All processes loaded state dict')
    
    return state_dict


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
