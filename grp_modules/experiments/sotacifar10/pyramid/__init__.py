def register(mf):

    # setting defaults for the absolute minimum setup, e.g. model, activations, normalization, ...
    mf.register_default_module("pyramidnet110", required_event="init_net")

    # get defaults of multistep
    mf.load("..multistep")

    # cifar+sgd traines best with multistep scheduler
    mf.register_default_module("multistep", required_event="scheduler_step", overwrite_globals={
        "scheduler.multistep.milestones": [150, 225],
    })

    # for some choices of an optimizer, we have better defaults
    mf.register_default_module("sgd", required_event="init_optimizer", overwrite_globals={
        "optimizer.sgd.nesterov": True,
    })

    # this is a default training/validate/testing experiment
    mf.register_default_module(["train", "validate", "test"], required_event="main", overwrite_globals={
        "main.epochs": 300,
    })
