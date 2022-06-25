import time

import numpy as np
import torch

from basicsr.utils.dist_util import init_dist, get_dist_info

init_dist('pytorch')
rank, world_size = get_dist_info()

print(f'{rank=}, {world_size=}')

INDEX = 10000
NELE = 1000
a = torch.rand(INDEX, NELE)
index = np.random.randint(INDEX-1, size=INDEX*8)
b = torch.from_numpy(index)

print(torch.__config__.parallel_info())
start = time.time()
for _ in range(100):
    res = a.index_select(0, b)
print(f"time: {time.time()-start}")