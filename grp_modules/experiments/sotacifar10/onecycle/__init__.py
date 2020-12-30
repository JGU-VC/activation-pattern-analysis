
def register(mf):

    mf.register_default_module("batchnorm", required_event="normalization_layer", overwrite_globals={
        "batchnorm.momentum": 0.05,
    })
    mf.register_default_module("onecyclelr", required_event="init_scheduler", overwrite_globals={
        "onecyclelr.anneal_strategy": lambda state, event: state["grp.scheduler.onecyclelr.AnnealStrategy"].COS,
        "onecyclelr.epochs_start": 0.1,
    })

    mf.load("..base")

    mf.register_default_module(["train", "validate", "test"], required_event="main", overwrite_globals={
        "main.epochs": 50,
        "batchsize": 256,
    })
    mf.register_default_module("sgd", required_event="init_optimizer", overwrite_globals={
        "sgd.momentum": 0.9,
        "sgd.weight_decay": 2e-4,
        "sgd.nesterov": True,

        "sgd.lr": 0.8,
        "onecyclelr.div_factor": 10,
        "onecyclelr.final_div_factor": 1e3,
        "onecyclelr.anneal_strategy": lambda state, event: state["grp.scheduler.onecyclelr.AnnealStrategy"].COS,
    })
    mf.register_default_module("adam", required_event="init_optimizer", overwrite_globals={
        "adam.lr": 1e-3,
        "adam.betas": (0.9, 0.99),
        "adam.weight_decay": 1e-2,
        "onecyclelr.div_factor": 1e10,
    })
