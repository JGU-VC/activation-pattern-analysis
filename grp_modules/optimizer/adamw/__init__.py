from torch.optim import AdamW as Optimizer


def register(mf):
    mf.load("optimizer")
    mf.event.register_optimizer_with_defaults(mf, Optimizer)
