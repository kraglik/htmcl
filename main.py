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

    with open('debug.cl', 'w') as f:
        f.write(prg_text)

    prg = cl.Program(ctx, prg_text)
    prg.build()

    return ctx, queue, prg


def main():
    ctx, queue, prg = build_ctx_queue_prg()

    heap = kma.build_kma(ctx, queue, prg, 2048)

    test_allocations = prg.test_allocations
    test_list_allocations = prg.test_list_allocations
    do_nothing = prg.do_nothing

    mf = cl.mem_flags

    ARRAY_SIZE = 10
    REPEATS_NUMBER = 10

    results = np.array([0] * ARRAY_SIZE, dtype=np.uint64)
    results_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=results)

    def do_nothing_exec():
        do_nothing(queue, results.shape, None, heap)
        queue.finish()

    def test_allocations_exec():
        test_allocations(queue, results.shape, (1, ), heap, results_b).wait()
        queue.finish()

    # test_allocations_exec()
    #
    # seconds_spent_doing_nothing = timeit.timeit(
    #     do_nothing_exec,
    #     number=REPEATS_NUMBER
    # )
    # seconds_spent_total = timeit.timeit(
    #     test_allocations_exec,
    #     number=REPEATS_NUMBER
    # )
    #
    # alloc_per_second = int(
    #     ARRAY_SIZE * REPEATS_NUMBER /
    #     (seconds_spent_total - seconds_spent_doing_nothing)
    # ) * 128
    #
    # print(f"{alloc_per_second} allocations per second")

    test_list_allocations(queue, results.shape, (1, ), heap, results_b).wait()

    cl.enqueue_copy(queue, results, results_b)
    print(results)

    # assert all(x == y for x, y in zip(results, list(range(0, ARRAY_SIZE * 10, 10)))), "Kernel returned wrong result"
    # print("Kernel allocations are OK")


if __name__ == '__main__':
    main()


