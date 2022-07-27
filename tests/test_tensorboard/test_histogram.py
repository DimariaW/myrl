import tensorboardX
import numpy as np

sw = tensorboardX.SummaryWriter(logdir="./log")
for i in range(100):
    sw.add_histogram("test", i + np.random.randn(1000), global_step=i)

for i in range(100):
    sw.add_histogram("test", -i + np.random.randn(1000), global_step=i)


