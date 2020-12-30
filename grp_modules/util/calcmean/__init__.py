import sys
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state, event):
        super().__init__()
        self.event = event
        self.state = state
        self.fc = nn.Linear(1, 1)
        self.welford = self.event.Welford()
        self.state["net"] = self

        event.optional.init_net_finished(self)

    def forward(self, net):
        count = np.prod([net.shape[s] for s in self.state["axes"]])
        mean = net.double().mean(self.state["axes"])
        var = net.double().var(self.state["axes"])
        self.welford.updateWithMeanVar(count, mean, var)
        return net.flatten(1)[:, :self.state["dataset.num_classes"]] * self.fc.weight


def after_epoch(state, event):
    del event  # unused
    torch.set_printoptions(precision=8, linewidth=120)
    print("dataset mean:", state["net"].welford.mean)
    print("dataset var:", state["net"].welford.var)
    print("dataset std:", state["net"].welford.std)
    sys.exit(0)


def register(mf):
    mf.load(["Welford", "noaugment"])
    mf.register_defaults({
        "axes": [0, 2, 3],
        "print_precision": 8
    })
    mf.overwrite_globals({
        "shuffle": False,
        "mean": [0, 0, 0],
        "std": [1, 1, 1]
    })
    mf.register_default_module('cpuloader', required_event='dataset', overwrite_globals={
        "with_transform": False,
        "val_prop": 0.0,
    })
    mf.register_event("init_net", Net, unique=True)
    mf.register_event("after_epoch", after_epoch)
