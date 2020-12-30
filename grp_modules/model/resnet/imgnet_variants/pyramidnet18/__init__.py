def register(mf):
    mf.register_default_module("basicblock_pyramid", required_event="resblock")
    mf.load("..pyramidnet_defaults")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_blocks": [2, 2, 2, 2],
    })
