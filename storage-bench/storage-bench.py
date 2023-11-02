import torch
import time
import os
import sys
import argparse

parser = argparse.ArgumentParser(description='BERT inference latency measurement')
parser.add_argument('--fp16', action='store_true', default=False, help='use fp16 for inference')
parser.add_argument('--scan', action='store_true', default=False, help='whether scan tensor size')
parser.add_argument('--tensor-size', type=int, nargs='+', default=[8, 2048, 2048], help='input tensor shape')

args = parser.parse_args()

# create tensor
tensor_size = tuple(args.tensor_size)

#  check device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print("GPU is unavailable, try CPU instead")
    device = torch.device('cpu')


def storage_benchmark(tensor_size: tuple, fp16_enabled: bool, display: bool = False):
    # create tensor
    print("#############################################") if display else None
    start_time = time.time()
    if fp16_enabled:
        tensor = torch.zeros(tensor_size, device=device, dtype=torch.half)
    else:
        tensor = torch.zeros(tensor_size, device=device)
    gpu_time = time.time() - start_time
    memory_size = tensor.element_size() * tensor.numel()
    print("Create tensor on GPU: {:<8f} sec, element_size: {:8f} MB".format(gpu_time, memory_size/1024/1024)) if display else None

    # GPU -> CPU
    start_time = time.time()
    tensor_cpu = tensor.to('cpu')
    gpu_to_cpu_time = time.time() - start_time
    print("Move tensor from GPU to CPU: {:<8f} sec".format(gpu_to_cpu_time)) if display else None

    # CPU -> DISK
    start_time = time.time()
    torch.save(tensor_cpu, 'tensor.pt')
    cpu_to_disk_time = time.time() - start_time
    print("Move tensor from CPU to disk: {:<8f} sec".format(cpu_to_disk_time)) if display else None

    # remove tensor from disk
    os.remove('tensor.pt')

# warm up
for i in range(3):
    storage_benchmark(tensor_size, args.fp16, display=False)

if args.scan:
    # scan from 16MB to 8GB
    for i in range(1, 11):
        tensor_size = tuple([pow(2,i), 2048, 2048])
        storage_benchmark(tensor_size, args.fp16, display=True)
else:
    storage_benchmark(tensor_size, args.fp16, display=True)