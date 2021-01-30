from htmcl.opencl import CLContext
from htmcl.layer import Layer
from htmcl.config import LayerConfig


def main():
    ctx = CLContext(interactive=False, debug=True)
    # ctx.run_test()

    layer = Layer(ctx, LayerConfig())
    print("layer on-GPU struct size is", layer.layer_struct_size(), "bytes")
    print("column on-GPU struct size is", layer.column_struct_size(), "bytes")
    print("cell on-GPU struct size is", layer.cell_struct_size(), "bytes")


if __name__ == '__main__':
    main()


