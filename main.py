from htmcl.opencl import CLContext
from htmcl.layer import Layer
from htmcl.htm import HTM
from htmcl.config import LayerConfig, RandomConnectionConfig


def main():
    ctx = CLContext(interactive=False, debug=True, heap_size_megabytes=256)

    htm = HTM(ctx)\
        .add_layer('i1',
                   LayerConfig(
                       is_input_layer=True,
                       layer_size=128))\
        .add_layer('l1', LayerConfig(layer_size=128))\
        .connect_layers('i1_to_l1',
                        'i1', 'l1',
                        RandomConnectionConfig(connection_probability=0.15))

    i1 = htm.get_layer('i1')
    l1 = htm.get_layer('l1')

    print("layer on-GPU struct size is", l1.layer_struct_size(), "bytes")
    print("column on-GPU struct size is", l1.column_struct_size(), "bytes")
    print("cell on-GPU struct size is", l1.cell_struct_size(), "bytes")


if __name__ == '__main__':
    main()
