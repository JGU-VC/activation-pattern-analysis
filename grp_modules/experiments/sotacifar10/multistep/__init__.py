def register(mf):
    mf.load("..base")

    # for some choices of an optimizer, we have better defaults
    mf.register_default_module("sgd", required_event="init_optimizer", overwrite_globals={
        "optimizer.sgd.lr": 0.1,
        "optimizer.sgd.weight_decay": 1e-4,
        "optimizer.sgd.momentum": 0.9,
    })
    mf.register_default_module("adam", required_event="init_optimizer", overwrite_globals={
        "optimizer.adam.lr": 3e-3,
        "optimizer.adam.weight_decay": 1e-4,
    })

    # cifar+sgd traines best with multistep scheduler
    mf.register_default_module("multistep", required_event="scheduler_step", overwrite_globals={
        "scheduler.multistep.milestones": [100, 150],
        "scheduler.multistep.gamma": 0.1,
    })
