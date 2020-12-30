from enum import Enum
from torch import nn


class ShortcutOption(Enum):
    A = 0
    B = 1


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def register(mf):
    mf.set_scope("...")
    mf.register_helpers({
        "ShortcutOption": ShortcutOption,
        "LambdaLayer": LambdaLayer,
    })
