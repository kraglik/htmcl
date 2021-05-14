import os
import pathlib
import time
import timeit

import pyopencl as cl
import numpy as np
import typing as t


class CLContext:

    KERNELS = [
        'kma.cl',
        'random.cl',
        'list.cl'
    ]

    HTM_KERNELS = [
        'htm_declarations.cl',
        'layer.cl',
        'proximal_dendrite.cl',
        'proximal_synapse.cl',
        'layer_connection.cl',
        'sdr.cl',
        'synapse.cl',
        'dendrite.cl',
        'cell.cl',
        'column.cl',
        'network.cl',
        'connection_routines.cl',
        'layer_preparation_routines.cl',
        'spatial_pooler.cl',
        'temporal_pooler.cl'
    ]

    MAIN_KERNELS = [
        'utils.cl'
    ]

    BLOCK_SIZE_BYTES = 4096
    CL_HEAP_SIZE_BYTES = 120
    MAX_U64 = 9223372036854775806

    def __init__(
            self,
            platform: t.Optional[int] = None,
            device: t.Optional[int] = None,
            interactive: bool = True,
            debug: bool = False,
            heap_size_megabytes: int = 256
    ):
        self.debug = debug
        self.heap_size_megabytes = heap_size_megabytes

        if platform is None or device is None:
            self.ctx = cl.create_some_context(interactive=interactive)

        else:
            platform = cl.get_platforms()[platform]
            device = platform.get_devices()[device]
            self.ctx = cl.Context([device])

        self.queue = cl.CommandQueue(self.ctx)
        self.mf = cl.mem_flags

        self.prg_text = ''

        self.prg_text += self.read_kernels(['kernels', 'utils'], self.KERNELS)
        self.prg_text += self.read_kernels(['kernels', 'htm'], self.HTM_KERNELS)
        self.prg_text += self.read_kernels(['kernels', 'utils'], self.MAIN_KERNELS)

        self.prg = cl.Program(self.ctx, self.prg_text)
        self.prg.build()
        self.heap = self.prepare_heap(self.heap_size_megabytes)
        self.randoms = self.create_random_seed_buffer(512_000)

        if self.debug:
            with open('debug.cl', 'w') as f:
                f.write(self.prg_text)

    def prepare_heap(self, heap_size_mbytes: t.Union[float, int]) -> cl.Buffer:
        n_blocks = (int(heap_size_mbytes * 1024 * 1024) - self.CL_HEAP_SIZE_BYTES) // self.BLOCK_SIZE_BYTES
        n_bytes = n_blocks * self.BLOCK_SIZE_BYTES + self.CL_HEAP_SIZE_BYTES

        self.heap = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=n_bytes)

        self.prg.clheap_init_step_1(
            self.queue,
            (1, ),
            None,
            self.heap,
            np.uint64(n_bytes)
        )
        self.queue.finish()

        for i in range(n_blocks, 1024):
            self.prg.clheap_init_step_2(
                self.queue,
                (1, ),
                None,
                self.heap,
                np.uint64(i),
                np.uint64(1024)
            )
            self.queue.finish()

        # self.run_unit_kernel(self.prg.clheap_init, self.heap, np.uint64(n_bytes))

        return self.heap

    def create_random_seed_buffer(self, n_seeds: int) -> cl.Buffer:
        randoms = np.random.randint(low=0, high=self.MAX_U64, size=n_seeds, dtype=np.uint64)
        self.randoms = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=randoms)

        return self.randoms

    def size_getter(self, getter: cl.Kernel) -> int:
        result_b = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=8)
        result = np.array([0], dtype=np.uint64)

        getter(self.queue, result.shape, None, result_b)
        cl.enqueue_copy(self.queue, result, result_b)
        self.queue.finish()

        return result[0]

    def make_buffer(self, size_bytes: int) -> cl.Buffer:
        return cl.Buffer(
            self.ctx,
            self.mf.READ_WRITE,
            size=int(size_bytes)
        )

    def run_unit_kernel(self, kernel, *args):
        kernel(self.queue, (1, ), None, *args)
        self.queue.finish()

    def run_test(self):
        test_allocations = self.prg.test_allocations
        test_list_allocations = self.prg.test_list_allocations
        test_random = self.prg.test_random
        do_nothing = self.prg.do_nothing

        array_size = 1664 * 4  # 1664 is the number of CUDA cores on GTX 970
        repeats_number = 10000

        results = np.array([0] * array_size, dtype=np.int32)
        randoms = np.random.randint(low=0, high=self.MAX_U64, size=array_size, dtype=np.uint64)
        random_floats = np.array([0] * array_size, dtype=np.float32)

        results_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=results)
        randoms_b = self.create_random_seed_buffer(array_size)
        random_floats_b = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=random_floats)

        def do_nothing_exec():
            do_nothing(self.queue, results.shape, None, self.heap)
            self.queue.finish()

        def test_allocations_exec():
            test_allocations(self.queue, results.shape, (1,), self.heap, results_b).wait()
            self.queue.finish()

        def test_random_exec():
            test_random(self.queue, randoms.shape, None, randoms_b, random_floats_b)
            self.queue.finish()

        def test_random_speed():
            seconds_spent_total = timeit.timeit(
                test_random_exec,
                number=repeats_number
            )
            randoms_per_second = int(
                array_size * repeats_number /
                seconds_spent_total
            ) * 128

            print(f"{randoms_per_second} random numbers per second")

        def test_allocation_speed():
            seconds_spent_doing_nothing = timeit.timeit(
                do_nothing_exec,
                number=repeats_number
            )
            seconds_spent_total = timeit.timeit(
                test_allocations_exec,
                number=repeats_number
            )

            alloc_per_second = int(
                array_size * repeats_number /
                (seconds_spent_total - seconds_spent_doing_nothing)
            ) * 128

            print(f"{alloc_per_second} allocations per second")

        def test_lists():
            test_list_allocations(self.queue, results.shape, (1,), self.heap, results_b).wait()

            cl.enqueue_copy(self.queue, results, results_b)
            print(results)

            assert all(
                x == y for x, y in zip(results, list(range(0, array_size * 10, 10)))), "Kernel returned wrong result"
            print("Kernel allocations are OK")

        test_random_speed()
        # test_allocation_speed()
        test_lists()

    @staticmethod
    def read_kernels(path: list, filenames: list) -> str:
        current_dir = pathlib.Path(__file__).parent.absolute()
        text = ''

        for file_name in filenames:
            with open(os.path.join(current_dir, *path, file_name), 'r') as f:
                cl_file = f.read()
                text += cl_file

        return text
