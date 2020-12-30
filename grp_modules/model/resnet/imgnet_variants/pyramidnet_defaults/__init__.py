
def register(mf):
    mf.load("...")
    mf.load("...blocks.util")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_planes": [64, 64 + 300],  # alpha=300
        "strides": [2, 2, 2, 2],
        "plane_increase_strategy": mf.state["PlaneIncreaseStrategy"].ADD,

        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        "replace_stride_with_dilation": [False, False, False],

        # first conv layer
        "first_conv_filters": 64,
        "first_conv_kernel_size": 7,
        "first_conv_padding": 3,
        "first_conv_stride": 2,
        "first_conv_activation": True,
        "first_conv_max_pool": True,

        # PyramidNets always use shortcut type A
        "option": mf.state["ShortcutOption"].A,
    }, scope="model.resnet")
