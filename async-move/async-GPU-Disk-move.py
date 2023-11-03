import time
import os
import torch
import torch.cuda
from torch.cuda import Stream
# import multiprocessing as mp
import threading

class Timer:
    def __init__(self):
        self.start = time.monotonic()

    def __call__(self):
        k = time.monotonic()
        v = k - self.start
        self.start = k
        return v

class CudaThread(threading.Thread):
    def __init__(self, stream, tensor, save_path):
        threading.Thread.__init__(self)
        self.stream = stream
        self.tensor = tensor
        self.save_path = save_path

    def run(self):
        # 在线程中执行CUDA操作
        with torch.cuda.stream(self.stream):
            self.save_tensor(self.tensor, self.save_path)

    def save_tensor(self, tensor, save_path):
        # start timer
        tim = Timer()
        # pin a tensor in CPU memory
        tensor_cpu_pinned = torch.empty(tensor.size(), pin_memory=True)
        print(f"Thread {threading.get_ident()}: pin cpu tensor: {tim()}")
        # create event
        ev = torch.cuda.Event()
        # async copy from GPU to pinned CPU memory
        tensor_cpu_pinned.copy_(tensor_gpu, non_blocking=True)
        print(f"Thread {threading.get_ident()}: async call return: {tim()}")
        ev.record()

        # Check if the event has finished
        self.stream.wait_event(ev)
        print(f"Thread {threading.get_ident()}: copy finish taks:{tim()}")
        torch.save(tensor_cpu_pinned, save_path)
        print(f"Thread {threading.get_ident()}: save to disk takes:{tim()}")

        self.stream.synchronize()

# 示例用法
if __name__ == '__main__':
    # create cuda stream
    CUDA_STREAM = torch.cuda.Stream()
    # create a tensor on GPU
    size=[32*8, 1024, 1024]
    tensor_gpu = torch.randn(size, device='cuda')
    memory_size = tensor_gpu.element_size() * tensor_gpu.numel() / 1024 / 1024
    print(f"tensor_gpu size:{memory_size}MB")

    # 指定保存路径
    save_path = 'tensor.pt'

    thread1 = CudaThread(CUDA_STREAM, tensor_gpu, save_path)

    main_timer = Timer()
    thread1.start()
    print(f"start thread takes:{main_timer()}")
    # cnt = 0
    # while thread1.is_alive():
    #     cnt += 1
    #     print(f"main thread cnt:{cnt}")
    thread1.join()
    print(f"finish thread takes:{main_timer()}")

    # check tensor
    tensor_cpu = torch.load(save_path)
    assert torch.equal(tensor_gpu.cpu(), tensor_cpu)
    os.remove(save_path)