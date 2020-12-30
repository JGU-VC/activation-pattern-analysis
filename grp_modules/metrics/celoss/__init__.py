from torch import nn


def loss(state, reduction="mean"):
    del state  # unused
    return nn.CrossEntropyLoss(reduction=reduction)


def register(mf):
    mf.set_scope("metrics")
    mf.register_event('init_loss', loss, unique=True)
