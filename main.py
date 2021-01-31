from htmcl.opencl import CLContext
from htmcl.layer import Layer
from htmcl.htm import HTM
from htmcl.config import LayerConfig, RandomConnectionConfig


def main():
    ctx = CLContext(interactive=False, debug=True)

    htm = HTM(ctx)\
        .add_layer('i1',
                   LayerConfig(
                       input_layer=True,
                       layer_size_x=1024,
                       layer_size_y=1))\
        .add_layer('l1', LayerConfig())\
        .connect_layers('i1', 'l1',
                        RandomConnectionConfig(connections_probability=0.2))

    i1 = htm.get_layer('i1')
    l1 = htm.get_layer('l1')

    print("layer on-GPU struct size is", l1.layer_struct_size(), "bytes")
    print("column on-GPU struct size is", l1.column_struct_size(), "bytes")
    print("cell on-GPU struct size is", l1.cell_struct_size(), "bytes")


if __name__ == '__main__':
    main()
