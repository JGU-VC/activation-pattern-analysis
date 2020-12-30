from enum import Enum
import torch
from miniflask import like


class AnnealStrategy(Enum):
    COS = 0
    LINEAR = 1


def after_init_optimizer(state, optimizer, net, parameters, *args, **kwargs):
    del net, parameters

    # calculate last index in case we are resuming training
    num_batches = state["main.num_batches"]
    last_batch_index = state["main.start_epoch"] * num_batches if state["main.start_epoch"] else -1
    state["scheduler"] = torch.optim.lr_scheduler.OneCycleLR(optimizer, state["optimizer.lr"], total_steps=None, epochs=state["overwrite_epochs"] if state["overwrite_epochs"] > 0 else state["epochs"], steps_per_epoch=num_batches + 1, pct_start=state["epochs_start"], anneal_strategy=state["anneal_strategy"].name.lower(), cycle_momentum=state["cycle_momentum"], base_momentum=state["base_momentum"], max_momentum=state["max_momentum"], div_factor=state["div_factor"], final_div_factor=state["final_div_factor"], last_epoch=last_batch_index)
    return optimizer, args, kwargs


def scheduler_step(state, event, *args, **kwargs):
    del event  # unused
    state["scheduler"].step()
    return None, args, kwargs


def register(mf):
    mf.register_defaults({
        "epochs_start": 0.1,
        "anneal_strategy": AnnealStrategy.COS,
        "cycle_momentum": lambda state, event: "optimizer.momentum" in state and state["optimizer.momentum"] > 0,
        "base_momentum": lambda state, event: 0.85 * state["optimizer.momentum"] if "optimizer.momentum" in state else 0,
        "max_momentum": lambda state, event: state["optimizer.momentum"] if "optimizer.momentum" in state else 0,
        "div_factor": 10,
        "final_div_factor": 1e4,
        "overwrite_epochs": like('main.epochs', alt=0),
    })
    mf.register_helpers({
        "AnnealStrategy": AnnealStrategy
    })
    mf.register_event('after_init_optimizer', after_init_optimizer, unique=False)
    mf.register_event('init_scheduler', after_init_optimizer)
    mf.register_event('scheduler_step', scheduler_step, unique=False)
    mf.register_event('after_step', scheduler_step, unique=False)
