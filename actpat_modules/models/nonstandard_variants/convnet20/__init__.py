def register(mf):
    mf.load("resnet20")
    mf.set_scope("grp.model.resnet")
    mf.overwrite_defaults({
        "short": False,
    }, scope="grp.model.resnet")
