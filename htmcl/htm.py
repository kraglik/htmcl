import pyopencl as cl
import numpy as np
import typing as t

from htmcl.connection import Connection
from htmcl.opencl import CLContext
from htmcl.layer import Layer
from htmcl.config import LayerConfig, RandomConnectionConfig
from htmcl.sdr import SDR


class HTM:
    def __init__(
            self,
            ocl: t.Optional[CLContext] = None,
            max_layers: int = 32,
            max_connections: int = 128
    ):
        self._ocl = ocl or CLContext()
        self._layers = []
        self._id_to_layer = {}
        self._name_to_layer = {}
        self._connections_by_name = {}
        self._connections = []

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

        self._add_layer_on_gpu(layer)

        self._next_layer_id += 1

        return self

    def _add_layer_on_gpu(self, layer: Layer):
        self._ocl.run_unit_kernel(
            self._ocl.prg.set_htm_layer,
            self._buffer,
            layer.buffer(),
            np.uint32(layer.id)
        )

    def connect_layers(
            self,
            connection_name: str,
            input_layer_name: str,
            target_layer_name: str,
            config: t.Union[RandomConnectionConfig]
    ) -> 'HTM':

        assert input_layer_name in self._name_to_layer,\
            f"Layer with name {input_layer_name} does not exist."

        assert target_layer_name in self._name_to_layer,\
            f"Layer with name {target_layer_name} does not exist."

        assert connection_name not in self._connections_by_name,\
            f"Connection with name {connection_name} already exist."

        if isinstance(config, RandomConnectionConfig):
            self._randomly_connect_layers(
                connection_name,
                self._name_to_layer[input_layer_name],
                self._name_to_layer[target_layer_name],
                config
            )

        return self

    def get_layer(self, layer_name):
        assert layer_name in self._name_to_layer,\
            f"Layer with name {layer_name} does not exist."

        return self._name_to_layer[layer_name]

    def _randomly_connect_layers(self, name: str, a: Layer, b: Layer, config: RandomConnectionConfig):
        assert config.connection_probability is not None,\
            "At least one 'connection_' parameter must be set"

        connection = self._create_connection(a, b, config)
        self._connections.append(connection)
        self._connections_by_name[name] = connection
        connection.connect(self._buffer)

    def _create_connection(self, i: Layer, o: Layer, config: RandomConnectionConfig) -> Connection:
        conn_id = self._next_connection_id
        self._next_connection_id += 1
        conn = Connection(self._ocl, i, o, conn_id, config)

        self._ocl.run_unit_kernel(
            self._ocl.prg.set_htm_connection,
            self._buffer,
            conn.get_buffer(),
            np.uint32(conn_id)
        )

        return conn

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

