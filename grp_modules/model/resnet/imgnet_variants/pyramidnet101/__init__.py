def register(mf):
    mf.register_default_module("bottleneck_pyramid", required_event="resblock")
    mf.load("..pyramidnet_defaults")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_blocks": [3, 4, 23, 3],
    })
