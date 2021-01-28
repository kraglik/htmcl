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

    prg_text = ""

    kernels = [
        'kma.cl',
        'random.cl',
        'list.cl',
        'htm.cl'
    ]

    for file_name in kernels:
        with open(os.path.join(current_dir, 'kernels', file_name), 'r') as f:
            cl_file = f.read()
            prg_text += cl_file

    with open('debug.cl', 'w') as f:
        f.write(prg_text)

    prg = cl.Program(ctx, prg_text)
    prg.build()

    return ctx, queue, prg


def main():
    ctx, queue, prg = build_ctx_queue_prg()

    heap = kma.build_kma(ctx, queue, prg, 1)

    test_allocations = prg.test_allocations
    test_list_allocations = prg.test_list_allocations
    test_random = prg.test_random
    do_nothing = prg.do_nothing

    mf = cl.mem_flags

    ARRAY_SIZE = 1024
    REPEATS_NUMBER = 1000

    results = np.array([0] * ARRAY_SIZE, dtype=np.int32)
    randoms = np.random.randint(low=0, high=9223372036854775806, size=ARRAY_SIZE, dtype=np.uint64)
    random_floats = np.array([0] * ARRAY_SIZE, dtype=np.float32)

    results_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=results)
    randoms_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=randoms)
    random_floats_b = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=random_floats)

    def do_nothing_exec():
        do_nothing(queue, results.shape, None, heap)
        queue.finish()

    def test_allocations_exec():
        test_allocations(queue, results.shape, (1, ), heap, results_b).wait()
        queue.finish()

    def test_randoms():
        avg, low, high = 0, 0, 0

        N_ITERS = 1_000

        for i in range(N_ITERS):
            test_random(queue, randoms.shape, (1, ), randoms_b, random_floats_b)
            cl.enqueue_copy(queue, random_floats, random_floats_b)

            avg += (sum(random_floats) / ARRAY_SIZE) / N_ITERS
            low = min(low, np.min(random_floats))
            high = max(high, np.max(random_floats))

        print(avg, low, high)

        queue.finish()

    def test_random_speed():
        seconds_spent_doing_nothing = timeit.timeit(
            do_nothing_exec,
            number=REPEATS_NUMBER
        )
        seconds_spent_total = timeit.timeit(
            lambda: test_random(queue, randoms.shape, None, randoms_b, random_floats_b).wait(),
            number=REPEATS_NUMBER
        )
        randoms_per_second = int(
            ARRAY_SIZE * REPEATS_NUMBER /
            (seconds_spent_total - seconds_spent_doing_nothing)
        ) * 128

        print(f"{randoms_per_second} random numbers second")

    def test_allocation_speed():
        seconds_spent_doing_nothing = timeit.timeit(
            do_nothing_exec,
            number=REPEATS_NUMBER
        )
        seconds_spent_total = timeit.timeit(
            test_allocations_exec,
            number=REPEATS_NUMBER
        )

        alloc_per_second = int(
            ARRAY_SIZE * REPEATS_NUMBER /
            (seconds_spent_total - seconds_spent_doing_nothing)
        ) * 128

        print(f"{alloc_per_second} allocations per second")

    # test_allocation_speed()

    test_random_speed()
    test_randoms()
    print(random_floats)

    test_list_allocations(queue, results.shape, (1, ), heap, results_b).wait()

    cl.enqueue_copy(queue, results, results_b)
    print(results)

    assert all(x == y for x, y in zip(results, list(range(0, ARRAY_SIZE * 10, 10)))), "Kernel returned wrong result"
    print("Kernel allocations are OK")


if __name__ == '__main__':
    main()


