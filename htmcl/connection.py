import numpy as np
import pyopencl as cl

from htmcl.config import RandomConnectionConfig
from htmcl.layer import Layer
from htmcl.opencl import CLContext


class Connection:
    def __init__(
            self,
            ocl: CLContext,
            input_layer: Layer,
            target_layer: Layer,
            id: int,
            config: RandomConnectionConfig
    ):
        self._ocl = ocl
        self.id = id
        self.input_layer = input_layer
        self.target_layer = target_layer
        self.input_layer_id = input_layer.id
        self.target_layer_id = target_layer.id
        self.config = config

        self._buffer = self._ocl.make_buffer(self._get_layer_connection_size_bytes())

        self._prepare_connection()

    def _get_layer_connection_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_layer_connection_size_bytes)

    def _prepare_connection(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.init_connection,
            self._buffer,
            np.uint32(self.input_layer_id),
            np.uint32(self.target_layer_id),
            np.float32(self.config.connection_probability)
        )

    def connect(self, htm_buffer: cl.Buffer, input: Layer, target: Layer):
        self._ocl.prg.randomly_connect_layers(
            self._ocl.queue,
            (int(self.target_layer.config.layer_size), ),
            (1, ),
            self._ocl.heap,
            htm_buffer,
            input.buffer(),
            target.buffer(),
            self._ocl.randoms,
            np.uint32(self.id),
            np.float32(self.config.connection_probability)
        )
        self._ocl.queue.finish()

    def get_buffer(self):
        return self._buffer
