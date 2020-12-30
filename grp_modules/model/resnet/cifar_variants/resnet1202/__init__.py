def register(mf):
    mf.load("..resnet_defaults")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_blocks": [200, 200, 200],
    })
