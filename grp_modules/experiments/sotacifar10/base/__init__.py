def register(mf):

    # this is a default training/validate/testing experiment
    mf.register_default_module(["train", "validate", "test"], required_event="main", overwrite_globals={
        "main.epochs": 200,
    })

    # although we are configuring for cifar10 here, this can also be overwritten
    mf.register_default_module("cifar10", required_event="dataset")

    # setting defaults for the absoulte minimum setup, e.g. model, activations, normalization, ...
    mf.register_default_module("resnet56", required_event="init_net")
    mf.register_default_module("batchnorm", required_event="normalization_layer")
    mf.register_default_module("relu", required_event="activation_layer")
    mf.register_default_module("conv2d", required_event="filter_layer")
    mf.register_default_module("classifier", required_event="classifier_layer")
    mf.register_default_module("optimizer", required_event="step")
    mf.register_default_module("gpu", required_event="send_net_to_device")

    # Augmentation
    mf.register_default_module("augment", required_event="dataset_transform", overwrite_globals={
        "data.augment.flip": True,
        "data.augment.cropsize": 32,
        "data.augment.croppadding": 4,
        "data.augment.rotationdeg": 0,
    })

    mf.overwrite_globals({
        "data.cpuloader.batchsize": 128,

        # Validation
        "data.cpuloader.val_prop": 0.01,
        "data.cpuloader.drop_last": False,
    })
