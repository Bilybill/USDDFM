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

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
# GPUS_PER_NODE = 8

# SETUP_RETRY_COUNT = 3


# def setup_dist():
#     """
#     Setup a distributed process group.
#     """
#     if dist.is_initialized():
#         return
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"
#     print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')

#     comm = MPI.COMM_WORLD
#     backend = "gloo" if not th.cuda.is_available() else "nccl"
#     print(f'use backend {backend}')

#     if backend == "gloo":
#         hostname = "localhost"
#     else:
#         hostname = socket.gethostbyname(socket.getfqdn())
#     os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
#     os.environ["RANK"] = str(comm.rank)
#     os.environ["WORLD_SIZE"] = str(comm.size)

#     port = comm.bcast(_find_free_port(), root=0)
#     os.environ["MASTER_PORT"] = str(port)
#     dist.init_process_group(backend=backend, init_method="env://")

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


# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if th.cuda.is_available():
#         return th.device(f"cuda")
#     return th.device("cpu")

def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return th.device(f"cuda:{local_rank}")
    return th.device("cpu")


# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across MPI ranks.
#     """
#     chunk_size = 2 ** 30  # MPI has a relatively small size limit
#     if MPI.COMM_WORLD.Get_rank() == 0:
#         with bf.BlobFile(path, "rb") as f:
#             data = f.read()
#         num_chunks = len(data) // chunk_size
#         if len(data) % chunk_size:
#             num_chunks += 1
#         MPI.COMM_WORLD.bcast(num_chunks)
#         for i in range(0, len(data), chunk_size):
#             MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
#     else:
#         num_chunks = MPI.COMM_WORLD.bcast(None)
#         data = bytes()
#         for _ in range(num_chunks):
#             data += MPI.COMM_WORLD.bcast(None)

#     return th.load(io.BytesIO(data), **kwargs)

# def load_state_dict(path, **kwargs):
#     """
#     Load a PyTorch file without redundant fetches across ranks.
#     """
#     try:
#         print(f'RANK {dist.get_rank()} loading state dict from {path}')
#         if dist.get_rank() == 0:
#             state_dict = th.load(path, **kwargs)
#         else:
#             state_dict = None
#         print(f'Loaded state dict from {path}')
        
#         # 广播主节点加载的状态字典到所有其他节点
#         if dist.get_world_size() > 1:
#             state_dict_list = [state_dict] if dist.get_rank() == 0 else [None]
#             dist.broadcast_object_list(state_dict_list, src=0)
#             state_dict = state_dict_list[0]
#         print(f'Broadcasted state dict to all nodes')
        
#         return state_dict
#     except Exception as e:
#         print(f'Error loading state dict: {e}')
#         raise e

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
