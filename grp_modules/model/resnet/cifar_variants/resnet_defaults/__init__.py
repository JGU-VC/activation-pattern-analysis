
def register(mf):
    mf.load("...")
    mf.load("...blocks.util")
    mf.set_scope("...")
    mf.register_default_module("basicblock_pre", required_event="resblock")
    mf.overwrite_defaults({
        "num_planes": [16, 32, 64],
        "strides": [1, 2, 2],
        "plane_increase_strategy": mf.state["PlaneIncreaseStrategy"].DEFAULT,

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

        # use shortcut option A for Cifar10 (as in original ResNet paper)
        "option": mf.state["ShortcutOption"].A,
    }, scope="model.resnet")
