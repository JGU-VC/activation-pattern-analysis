from torch.nn import BatchNorm2d as Normalization


def register(mf):
    mf.load("normalization")
    mf.event.register_normalization_with_defaults(mf, Normalization)
