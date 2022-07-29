import time
import torch


def process(data):
    data = torch.from_numpy(data)

    #data.share_memory_()
    return data


def test_process(queue):

    for i in range(20):

        tensor1 = torch.randn(512, 512, 51)
        beg = time.time()
        #tensor1.share_memory_()
        mid = time.time()
        queue.put(tensor1)
        end = time.time()
        print(f"share time:{mid-beg}, put time:{end-mid}")
