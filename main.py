import os
import pathlib
import timeit

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

    heap = kma.build_kma(ctx, queue, prg, 1024)

    test_allocations = prg.test_allocations
    do_nothing = prg.do_nothing

    mf = cl.mem_flags

    ARRAY_SIZE = 512
    REPEATS_NUMBER = 10000

    results = np.array([0] * ARRAY_SIZE, dtype=np.int32)
    results_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=results)

    seconds_spent_doing_nothing = timeit.timeit(
        lambda: do_nothing(queue, results.shape, None, heap),
        number=REPEATS_NUMBER
    )
    seconds_spent_total = timeit.timeit(
        lambda: test_allocations(queue, results.shape, None, heap, results_b),
        number=REPEATS_NUMBER
    )

    alloc_per_second = int(
        ARRAY_SIZE * REPEATS_NUMBER /
        (seconds_spent_total - seconds_spent_doing_nothing)
    )

    print(f"{alloc_per_second} allocations per second")

    cl.enqueue_copy(queue, results, results_b)

    assert all(x == y for x, y in zip(results, list(range(0, -ARRAY_SIZE, -1)))), "Kernel returned wrong result"
    print("Kernel allocations are OK")


if __name__ == '__main__':
    main()


