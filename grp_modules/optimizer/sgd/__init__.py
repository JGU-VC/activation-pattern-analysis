from torch.optim import SGD as Optimizer


def register(mf):
    mf.load("optimizer")
    mf.event.register_optimizer_with_defaults(mf, Optimizer)
