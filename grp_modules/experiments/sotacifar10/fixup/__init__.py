def register(mf):

    mf.load("..multistep")
    mf.load("nonorm")
    mf.load("basicblock_fixup")

    mf.register_default_module("sgd", required_event="init_optimizer", overwrite_globals={
        "optimizer.sgd.weight_decay": 5e-4,
    })
