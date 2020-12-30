def register(mf):
    mf.load("..resnet_defaults")
    mf.set_scope("...")
    mf.register_defaults({
        "groups": 1,
        "width_per_group": 64,
        "num_blocks": [3, 4, 23, 3],
    })
