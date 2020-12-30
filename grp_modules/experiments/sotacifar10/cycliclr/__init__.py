def register(mf):

    mf.load("..base")
    mf.register_default_module("cycliclr", required_event="init_scheduler", overwrite_globals={
        "cycliclr.epochs_up": 10
    })

    mf.register_default_module("sgd", required_event="init_optimizer", overwrite_globals={
        "optimizer.sgd.lr": 0.2,
        "optimizer.sgd.weight_decay": 5e-4,
        "optimizer.sgd.momentum": 0.9,
        "optimizer.sgd.nesterov": False,
    })

    mf.register_default_module(["train", "validate", "test"], required_event="main", overwrite_globals={
        "main.epochs": 100,
    })

    mf.overwrite_globals({
        "data.cpuloader.batchsize": 128,
    })
