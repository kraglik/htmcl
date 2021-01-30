import numpy as np
import pyopencl as cl

from htmcl.config import LayerConfig
from htmcl.opencl import CLContext
from htmcl.sdr import SDR


class Layer:
    def __init__(self, ocl: CLContext, config: LayerConfig):
        self.ocl = ocl
        self.config = config

        self._layer_struct_size = self._get_layer_size_bytes()
        self._cell_struct_size = self._get_cell_size_bytes()
        self._column_struct_size = self._get_column_size_bytes()

        self._buffer = cl.Buffer(
            self.ocl.ctx,
            self.ocl.mf.READ_WRITE,
            size=self._layer_struct_size
        )

        if self.config.input_layer:
            self._cells_buffer = None
            self._columns_buffer = None

        else:
            self._cells_buffer = self.ocl.make_buffer(size_bytes=self._get_cells_buffer_size())
            self._columns_buffer = self.ocl.make_buffer(size_bytes=self._get_columns_buffer_size())

        self._sdr = SDR(
            ocl,
            config.layer_size_x,
            config.layer_size_y,
            config.cells_per_column
        )

    def _get_layer_size_bytes(self) -> int:
        return self.ocl.size_getter(getter=self.ocl.prg.get_layer_size_bytes)

    def _get_column_size_bytes(self) -> int:
        return self.ocl.size_getter(getter=self.ocl.prg.get_column_size_bytes)

    def _get_cell_size_bytes(self) -> int:
        return self.ocl.size_getter(getter=self.ocl.prg.get_cell_size_bytes)

    def _get_cells_buffer_size(self) -> int:
        return int(
            self._cell_struct_size * (
                self.config.layer_size_x *
                self.config.layer_size_y *
                self.config.cells_per_column
            )
        )

    def _get_columns_buffer_size(self) -> int:
        return int(
            self._column_struct_size * (
                self.config.layer_size_x *
                self.config.layer_size_y
            )
        )

    def layer_struct_size(self) -> int:
        return self._layer_struct_size

    def column_struct_size(self) -> int:
        return self._column_struct_size

    def cell_struct_size(self) -> int:
        return self._cell_struct_size


