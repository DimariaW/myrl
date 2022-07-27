from myrl.memory import MemoryReplay, PriorityMemoryReplay
import numpy as np

mr = MemoryReplay(maxlen=500, batch_size=32, cached_in_device=False)

pmr = PriorityMemoryReplay(500, batch_size=32, cached_in_device=False)

for _ in range(100):
    pmr.cache(np.random.randn(10), 1, np.random.randn(10), 0, False)

s, a, s_, r, d, p, i = pmr.recall()
pmr.update_priority(i, p+1)
print("------")
