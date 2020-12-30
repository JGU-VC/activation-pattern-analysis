def register(mf):
    mf.load("..resnet_defaults")
    mf.set_scope("...")
    mf.register_defaults({
        "groups": 32,
        "width_per_group": 8,
        "num_blocks": [3, 4, 23, 3],
    })
