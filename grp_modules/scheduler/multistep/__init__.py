import torch


def after_init_optimizer(state, optimizer, net, parameters, *args, **kwargs):
    del net, parameters  # unused
    state["scheduler"] = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=state["milestones"], last_epoch=state["main.start_epoch"] - 1, gamma=state["gamma"])
    return optimizer, args, kwargs


def scheduler_step(state, *args, **kwargs):
    del args, kwargs  # unused
    state["scheduler"].step()


def register(mf):
    mf.register_defaults({
        "milestones": [10, 20],
        "gamma": 0.1,
    })
    mf.register_event('after_init_optimizer', after_init_optimizer, unique=False)
    mf.register_event('init_scheduler', after_init_optimizer)
    mf.register_event('after_epoch', scheduler_step, unique=False)
    mf.register_event('scheduler_step', scheduler_step, unique=False)
