import time

import torch

class Timer:
    def __init__(self):
        self.start = time.monotonic()

    def __call__(self):
        k = time.monotonic()
        v = k - self.start
        self.start = k
        return v

if __name__ == '__main__':


    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        # print(torch.cuda.current_stream())

        # x = torch.ones((32, 256, 220, 220), pin_memory=True)
        tim = Timer()  
        t1 = time.time()              
        c = torch.empty((10000, 10000), device='cuda')
        print(tim())
        # x = x.to(torch.device('cuda'), non_blocking=True)
        # print(tim())
        # c[0, :, :, :, :].copy_(x, non_blocking=True)
        # print(tim())

        # t = (x.min() - x.max()).to(torch.device("cpu"), non_blocking=True)

        t = c
        print('mark0', tim())
        t = t.to('cpu', non_blocking=True)
        print('mark', tim())
        ev = torch.cuda.Event()
        ev.record()
        t2 = time.time()              

    # print(torch.cuda.current_stream())
    print(tim())
    # print(t)
    ev.synchronize()
    print(tim())
    # print(t)