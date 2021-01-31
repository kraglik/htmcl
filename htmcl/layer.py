import numpy as np
import pyopencl as cl

from htmcl.config import LayerConfig
from htmcl.opencl import CLContext
from htmcl.sdr import SDR


class Layer:
    def __init__(self, ocl: CLContext, layer_id: int, config: LayerConfig):
        self.id = layer_id
        self._ocl = ocl
        self.config = config

        self._layer_struct_size = self._get_layer_size_bytes()
        self._cell_struct_size = self._get_cell_size_bytes()
        self._column_struct_size = self._get_column_size_bytes()

        self._buffer = cl.Buffer(
            self._ocl.ctx,
            self._ocl.mf.READ_WRITE,
            size=self._layer_struct_size
        )

        if self.config.input_layer:
            self._cells_buffer = None
            self._columns_buffer = None

        else:
            self._cells_buffer = self._ocl.make_buffer(self._get_cells_buffer_size())
            self._columns_buffer = self._ocl.make_buffer(self._get_columns_buffer_size())

        self._sdr = SDR(
            ocl,
            config.layer_size_x,
            config.layer_size_y,
            config.cells_per_column
        )
        self._prepare_buffers()
        self._prepare_coefficients()

    def _prepare_buffers(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.prepare_layer_buffers,
            self._buffer,
            self._cells_buffer,
            self._columns_buffer,
            self._sdr.buffer()
        )

    def _prepare_coefficients(self):
        self._prepare_primary_coefficients()
        self._prepare_boost_coefficients()
        self._prepare_segment_coefficients()

    def _prepare_primary_coefficients(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.prepare_layer_primary_coefficients,
            self._buffer,
            np.uint32(self.config.layer_size_x),
            np.uint32(self.config.layer_size_y),
            np.uint32(self.config.cells_per_column),
            np.uint32(self.config.learning),
            np.uint32(self.id)
        )

    def _prepare_boost_coefficients(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.prepare_layer_boost_coefficients,
            self._buffer,
            np.float32(self.config.boost_increase),
            np.float32(self.config.boost_decrease),
            np.float32(self.config.boost_strength),
        )

    def _prepare_segment_coefficients(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.prepare_layer_segment_coefficients,
            self._buffer,
            np.uint32(self.config.segment_activation_threshold),
            np.uint32(self.config.segment_minimal_threshold),
            np.uint32(self.config.initial_synapses_per_apical_segment),
            np.uint32(self.config.initial_synapses_per_distal_segment),
            np.uint32(self.config.apical_segments_per_cell),
            np.uint32(self.config.distal_segments_per_cell)
        )

    def _get_layer_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_layer_size_bytes)

    def _get_column_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_column_size_bytes)

    def _get_cell_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_cell_size_bytes)

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
