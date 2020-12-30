
def register(mf):
    mf.load("...")
    mf.load("...blocks.util")
    mf.set_scope("...")
    mf.register_default_module("bottleneck_pre", required_event="resblock")
    mf.overwrite_defaults({
        "num_planes": [64, 128, 256, 512],
        "strides": [2, 2, 2, 2],
        "plane_increase_strategy": mf.state["PlaneIncreaseStrategy"].DEFAULT,

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

        # use shortcut option B for ImageNet (as in original ResNet paper)
        "option": mf.state["ShortcutOption"].B
    }, scope="model.resnet")
