import time
import torch


def process(data):
    time.sleep(0.1)
    return data


def test_process(queue):
    for i in range(20):
        tensor1 = torch.randn(512, 512, 52)
        time.sleep(0.01)
        queue.put(tensor1)
