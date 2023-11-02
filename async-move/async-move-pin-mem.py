import torch
import time

# create a tensor on GPU
size=[128*4, 1024, 1024]
tensor_gpu = torch.randn(size, device='cuda')
memory_size = tensor_gpu.element_size() * tensor_gpu.numel() / 1024 / 1024

print(f"tensor_gpu size:{memory_size}MB")
# pin a tensor in CPU memory
tensor_cpu_pinned = torch.empty(size, pin_memory=True)

# copy tensor from GPU to CPU in a non-blocking way
start_time = time.time()
tensor_cpu_pinned.copy_(tensor_gpu, non_blocking=True)

# wait for the copy to complete
tmp_time = time.time()
torch.cuda.synchronize()
end_time = time.time()

print("before sync time:", tmp_time - start_time)
# print the tensors
# print("GPU tensor:\n", tensor_gpu)
# print("CPU pinned tensor:\n", tensor_cpu_pinned)

# print the time taken for the copy operation
print("Time taken for the copy operation:", end_time - start_time, "seconds")
