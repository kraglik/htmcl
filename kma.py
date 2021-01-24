import numpy as np
import pyopencl as cl


BLOCK_SIZE_BYTES = 4 * (2 ** 10)
CL_HEAP_SIZE_BYTES = 120


def build_kma(ctx, queue, prg, n_blocks) -> cl.Buffer:
    clheap_init = prg.clheap_init
    n_bytes = n_blocks * BLOCK_SIZE_BYTES + CL_HEAP_SIZE_BYTES

    mf = cl.mem_flags

    heap = cl.Buffer(ctx, mf.READ_WRITE, size=n_bytes)

    clheap_init(queue, (1, ), None, heap, np.uint64(n_bytes))

    return heap


