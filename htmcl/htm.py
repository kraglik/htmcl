import pyopencl as cl
import numpy as np
import typing as t

from htmcl.opencl import CLContext
from htmcl.layer import Layer
from htmcl.config import LayerConfig, RandomConnectionConfig
from htmcl.sdr import SDR


class HTM:
    def __init__(
            self,
            ocl: t.Optional[CLContext] = None,
            max_layers: int = 128,
            max_connections: int = 512
    ):
        self._ocl = ocl or CLContext()
        self._layers = []
        self._id_to_layer = {}
        self._name_to_layer = {}

        self._next_layer_id = 0
        self._next_connection_id = 0

        self._max_layers = max_layers
        self._max_connections = max_connections

        self._buffer = self._ocl.make_buffer(self._get_htm_size_bytes())
        self._layers_buffer = self._ocl.make_buffer(self._get_layer_pointers_buffer_size_bytes())
        self._connections_buffer = self._ocl.make_buffer(self._get_connections_buffer_size_bytes())

        self._prepare_buffers()

    def add_layer(
            self,
            layer_name: str,
            layer_config: LayerConfig
    ) -> 'HTM':

        assert layer_name not in self._name_to_layer, \
            f"Layer with name {layer_name} already exist"

        layer = Layer(self._ocl, self._next_layer_id, layer_config)

        self._id_to_layer[self._next_layer_id] = layer
        self._name_to_layer[layer_name] = layer

        self._next_layer_id += 1

        return self

    def connect_layers(
            self,
            input_layer_name: str,
            target_layer_name: str,
            config: t.Union[RandomConnectionConfig]
    ) -> 'HTM':

        assert input_layer_name in self._name_to_layer,\
            f"Layer with name {input_layer_name} does not exist."

        assert target_layer_name in self._name_to_layer,\
            f"Layer with name {target_layer_name} does not exist."

        if isinstance(config, RandomConnectionConfig):
            self._randomly_connect_layers(
                self._name_to_layer[input_layer_name],
                self._name_to_layer[target_layer_name],
                config
            )

        return self

    def get_layer(self, layer_name):
        assert layer_name in self._name_to_layer,\
            f"Layer with name {layer_name} does not exist."

        return self._name_to_layer[layer_name]

    def _randomly_connect_layers(self, a: Layer, b: Layer, config: RandomConnectionConfig):
        assert config.connections_count is not None or config.connection_probability is not None,\
            "At least one 'connection_' parameter must be set"

        connection_id = self._create_connection()

    def _create_connection(self) -> int:
        pass

    def _get_htm_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_htm_size_bytes)

    def _get_layer_connection_size_bytes(self) -> int:
        return self._ocl.size_getter(getter=self._ocl.prg.get_layer_connection_size_bytes)

    def _get_layer_pointers_buffer_size_bytes(self) -> int:
        return 8 * self._max_layers

    def _get_connections_buffer_size_bytes(self) -> int:
        return self._max_connections * self._get_layer_connection_size_bytes()

    def _prepare_buffers(self):
        self._ocl.run_unit_kernel(
            self._ocl.prg.init_htm,
            self._buffer,
            self._layers_buffer,
            self._connections_buffer
        )

