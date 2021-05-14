import numpy as np
import pyopencl as cl

from htmcl.opencl import CLContext


class SDR:
    def __init__(
            self,
            ocl: CLContext,
            size: int,
    ):
        self.ocl = ocl
        self.size = size
        self._sdr_struct_size = self._get_sdr_size_bytes()
        self._buffer = cl.Buffer(ocl.ctx, ocl.mf.READ_WRITE, size=self._sdr_struct_size)

        self._init_sdr()

    def _get_sdr_size_bytes(self):
        result_b = cl.Buffer(self.ocl.ctx, self.ocl.mf.READ_WRITE, size=8)
        result = np.array([0], dtype=np.uint64)

        self.ocl.run_unit_kernel(
            self.ocl.prg.get_sdr_size_bytes,
            result_b,
            np.uint32(self.size)
        )
        cl.enqueue_copy(self.ocl.queue, result, result_b)
        self.ocl.queue.finish()

        return result[0]

    def sdr_struct_size(self):
        return self._sdr_struct_size

    def _init_sdr(self):
        self.ocl.run_unit_kernel(
            self.ocl.prg.init_sdr,
            self._buffer,
            np.uint32(self.size)
        )

    def buffer(self):
        return self._buffer

