from enum import Enum
from torch import nn
import torch


class DryRunMode(Enum):
    DUMMYNET = 0
    DUMMYDATA = 1
    FULL = 2


class Net_on_data(nn.Module):
    def __init__(self, state, event):
        super().__init__()
        self.event = event
        self.state = state
        self.fc = nn.Linear(1, 1)

        event.optional.init_net_finished(self)

    def forward(self, net):
        return net.flatten(1)[:, :self.state["dataset.num_classes"]] * self.fc.weight


class DummyData:

    """Iterator that counts upward forever."""

    def __init__(self, num, batch_size, num_classes, shape=None):
        self.i = 0
        self.num = num
        self.batch_size = batch_size
        self.num_classes = num_classes

        if shape is not None:
            self.tensor = torch.FloatTensor(torch.Size(shape))
        else:
            self.tensor = torch.FloatTensor(torch.Size([self.batch_size, self.num_classes]))
        self.out = torch.ones(torch.Size([self.batch_size]), dtype=torch.long)

    def __iter__(self):
        self.i = 0
        return self

    def __len__(self):
        return self.num

    def __next__(self):
        if self.i < self.num:
            self.i += 1
        else:
            raise StopIteration
        return self.tensor, self.out


def before_training(state):
    if state["drymode"].value >= DryRunMode.DUMMYDATA.value:
        if state["drymode"] == DryRunMode.DUMMYDATA:
            x = next(iter(state["trainloader"]))
            shape = x[0].shape
        else:
            shape = None
        state["trainloader"] = DummyData(state["num_batches"], state["batchsize"], state["num_classes"], shape=shape)


def init(state, event):

    # replace the network independent of loaded modules
    # (  this uses internal miniflask magic
    #    -> to be replaced with the child modules feature (see https://github.com/da-h/miniflask/issues/20) )
    if state["drymode"] == DryRunMode.DUMMYNET or state["drymode"] == DryRunMode.FULL:
        event.init_net  # pylint: disable=pointless-statement
        del event.init_net  # pylint: disable=pointless-statement
        del state["mf"].event_objs["init_net"]
        state["mf"].register_event("init_net", Net_on_data, unique=True)


def register(mf):
    mf.register_defaults({
        "drymode": DryRunMode
    })
    mf.register_helpers({
        "mf": mf
    })
    mf.register_event("init", init, unique=False)
    mf.register_event("before_training", before_training, unique=False)
