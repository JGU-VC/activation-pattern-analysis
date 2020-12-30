def register(mf):
    mf.load("resnet20")
    mf.set_scope("grp.model.resnet")
    mf.register_default_module("basicblock_pre", required_event="resblock")
    mf.overwrite_defaults({
        "short": False,
        "zero_init_residual": False,
        "first_conv_filters": 32,
        "num_planes": [32, 32, 32],
    }, scope="grp.model.resnet")
