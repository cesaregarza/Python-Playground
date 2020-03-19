import random, math
import numpy as np
from numba import jit, int64


def _rand_below(n):
    k = n.bit_length()
    r = random.getrandbits(k)
    while r >= n:
        r = random.getrandbits(k)
    return r

@jit('int64[:](int64[:], int64)',nopython=True)
def sample(li, k):
    n = len(li)
    result = np.empty(k, dtype=np.int64)

    selected = set()
    for i in range(k):
        j = np.random.randint(n)
        while j in selected:
            j = np.random.randint(n)
        selected.add(j)
        result[i] = li[j]
    return result