import os
import pathlib

import numpy as np
import pyopencl as cl

import kma


def build_ctx_queue_prg():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    current_dir = pathlib.Path(__file__).parent.absolute()

    with open(os.path.join(current_dir, 'kernels', 'kma.cl'), 'r') as f:
        cl_kma = f.read()

    with open(os.path.join(current_dir, 'kernels', 'htm.cl'), 'r') as f:
        cl_htm = f.read()

    prg_text = cl_kma + cl_htm

    prg = cl.Program(ctx, prg_text)
    prg.build()

    return ctx, queue, prg


def main():
    ctx, queue, prg = build_ctx_queue_prg()

    heap = kma.build_kma(ctx, queue, prg, 64)

    test_allocations = prg.test_allocations

    mf = cl.mem_flags

    results = np.array([0] * 10, dtype=np.int32)
    results_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=results)

    test_allocations(queue, results.shape, None, heap, results_b)

    cl.enqueue_copy(queue, results, results_b)

    assert all(x == y for x, y in zip(results, list(range(0, -10, -1)))), "Kernel returned wrong result"
    print("Kernel allocations are OK")


if __name__ == '__main__':
    main()


