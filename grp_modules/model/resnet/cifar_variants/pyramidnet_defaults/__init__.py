
def register(mf):
    mf.load("...")
    mf.load("...blocks.util")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_planes": [16, 16 + 84],  # alpha=84
        "strides": [1, 2, 2],
        "plane_increase_strategy": mf.state["PlaneIncreaseStrategy"].ADD,

        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        "replace_stride_with_dilation": [False, False],

        # first conv layer
        "first_conv_activation": False,
        "first_conv_filters": 16,
        "first_conv_kernel_size": 3,
        "first_conv_padding": 1,
        "first_conv_stride": 1,
        "first_conv_max_pool": False,

        # PyramidNets always use shortcut type A
        "option": mf.state["ShortcutOption"].A,
    }, scope="model.resnet")
