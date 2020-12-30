import torch
from colored import fg, attr
import numpy as np


def dataloader(state, event, subset, use_cache=True, deterministic=False, with_transform=None):
    loader_name = "%sloader" % subset
    if loader_name in state and use_cache:
        return state[loader_name]

    print(attr('dim'), end='')
    trainset = event.dataset("train", with_transform=state["with_transform"] if with_transform is None else with_transform)
    testset = event.dataset("test", with_transform=state["with_transform"] if with_transform is None else with_transform)
    valset = event.dataset("val", with_transform=state["with_transform"] if with_transform is None else with_transform)
    trainset_size = len(trainset)
    valset_size = len(valset) if valset else 0
    testset_size = len(testset) if testset else 0
    print(attr('reset'), end='')
    valloader = None  # pylint: disable=possibly-unused-variable
    trainloader = None  # pylint: disable=possibly-unused-variable
    testloader = None  # pylint: disable=possibly-unused-variable

    # dis-/enable multiprocessin
    num_workers = state["num_workers"]
    if state["disable_multiprocessing"]:
        num_workers = 0

    # set seed for all data loader workers
    def worker_init(worker_id):
        del worker_id  # unused
        event.optional.set_seed()

    # split train in train and val if val_prop != 0 and there is no valset
    if valset is None and state["val_prop"] > 0 and hasattr(event, "validate"):
        valset_size = int(np.floor(trainset_size * state["val_prop"]))
        trainset_size -= valset_size
        trainset, valset = torch.utils.data.random_split(
            trainset,
            [trainset_size, valset_size]
        )
        print((attr('dim') + "Splitted %d %% of trainset as valset" + attr('reset')) % int(state["val_prop"] * 100))

    # add surplus of data to validation set instead of throwing away
    surplus = 0
    if state["drop_last"]:
        surplus = trainset_size - trainset_size // state["batchsize"] * state["batchsize"]
        trainset_size -= surplus
        valset_size += surplus

        print("  -> " + attr('bold') + fg('green') + "batches.drop_last" + attr('reset') + " moved %d samples to Valset." % surplus)

    # only use data_subset train_data
    if state["data_subset"] != 0:
        kwargs = {}
        if "seed" in state:
            kwargs = {"generator": torch.Generator().manual_seed(state["seed"])}
        trainset, _ = torch.utils.data.random_split(
            trainset,
            [state["data_subset"], trainset_size - state["data_subset"] + surplus],
            **kwargs
        )
        trainset_size = state["data_subset"]
        print("  -> " + fg('red') + "Using subset of %s data points." % (attr('bold') + str(state["data_subset"]) + attr('reset') + fg('red')), attr('reset'))

        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False if deterministic else state["shuffle"], batch_size=state["batchsize"],
                                                  num_workers=num_workers, drop_last=state["drop_last"],
                                                  worker_init_fn=worker_init, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False if deterministic else state["shuffle"], batch_size=state["batchsize"],
                                                  num_workers=num_workers, drop_last=state["drop_last"],
                                                  worker_init_fn=worker_init, pin_memory=True,
                                                  sampler=None)

    if valset is not None:
        valloader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=state["batchsize"],
                                                num_workers=num_workers,
                                                worker_init_fn=worker_init, pin_memory=True,
                                                sampler=None)

    testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=state["batchsize"],
                                             num_workers=num_workers,
                                             worker_init_fn=worker_init, pin_memory=True)

    print((attr('dim') + "Trainset size: %d. Valset size: %d. Testset size: %d" + attr('reset')) % (trainset_size, valset_size, testset_size))

    if use_cache:
        for _subset in ["train", "test", "val"]:
            loader_name = "%sloader" % _subset
            state[loader_name] = locals()["%sloader" % _subset]
    return locals()["%sloader" % subset]


def register(mf):
    mf.register_defaults({
        "batchsize": 128,
        "batchsize_test": 128,
        "data_subset": 0,
        "drop_last": True,
        "val_prop": 0.05,
        "shuffle": True,
        "num_workers": torch.get_num_threads(),
        "disable_multiprocessing": False,
        "with_transform": True,
    })
    mf.register_event("dataloader", dataloader, unique=True)
