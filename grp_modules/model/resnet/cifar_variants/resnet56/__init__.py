def register(mf):
    mf.load("..resnet_defaults")
    mf.set_scope("...")
    mf.overwrite_defaults({
        "num_blocks": [9, 9, 9],
    })
