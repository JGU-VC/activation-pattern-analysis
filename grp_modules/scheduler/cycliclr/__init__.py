import torch


def after_init_optimizer(state, optimizer, net, parameters, *args, **kwargs):
    del net, parameters

    # calculate last index in case we are resuming training
    num_batches = state["main.num_batches"]
    if state["main.start_epoch"] > 0:
        last_batch_index = state["main.start_epoch"] * num_batches
    else:
        last_batch_index = -1

    step_size_up = int(state["epochs_up"] * state["main.num_batches"])
    step_size_down = int(state["epochs_down"] * state["main.num_batches"]) if state["epochs_down"] >= 0 else None

    state["scheduler"] = torch.optim.lr_scheduler.CyclicLR(optimizer, state["base_learning_rate_factor"] * state["optimizer.lr"], max_lr=state["optimizer.lr"], mode=state["policy"], cycle_momentum=state["cycle_momentum"], base_momentum=state["base_momentum"], step_size_up=step_size_up, step_size_down=step_size_down, max_momentum=state["max_momentum"], last_epoch=last_batch_index)
    return optimizer, args, kwargs


def scheduler_step(state, *args, **kwargs):
    state["scheduler"].step()
    return None, args, kwargs


def register(mf):
    mf.register_defaults({
        "policy": 'triangular2',  # also possible: triangular, exp_range
        "max_momentum": lambda state, event: state["optimizer.momentum"] if "optimizer.momentum" in state else 0,
        "base_momentum": lambda state, event: 0.85 * state["optimizer.momentum"] if "optimizer.momentum" in state else 0,
        "cycle_momentum": lambda state, event: "optimizer.momentum" in state and state["optimizer.momentum"] > 0,
        "base_learning_rate_factor": 0.01,
        "epochs_up": 10.0,
        "epochs_down": -1,  # same as up
    })
    mf.register_event('after_init_optimizer', after_init_optimizer, unique=False)
    mf.register_event('init_scheduler', after_init_optimizer)
    mf.register_event('after_step', scheduler_step, unique=False)
    mf.register_event('scheduler_step', scheduler_step, unique=False)
