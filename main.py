from htmcl.opencl import CLContext


def main():
    ctx = CLContext(interactive=False, debug=True)
    ctx.run_test()


if __name__ == '__main__':
    main()


