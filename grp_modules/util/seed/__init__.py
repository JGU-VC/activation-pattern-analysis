import random
import torch
import numpy as np


def set_seed(state):
    seed = state["seed"]
    if seed >= 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.benchmark = True


def rng_save(state):
    state["rng_state_gpu_torch"] = torch.cuda.get_rng_state_all()
    state["rng_state_cpu_torch"] = torch.get_rng_state()
    state["rng_state_cpu_numpy"] = np.random.get_state()
    state["rng_state_cpu_random"] = random.getstate()


def rng_restore(state):
    torch.cuda.set_rng_state_all(state["rng_state_gpu_torch"])
    torch.set_rng_state(state["rng_state_cpu_torch"])
    np.random.set_state(state["rng_state_cpu_numpy"])
    random.setstate(state["rng_state_cpu_random"])


def register(mf):
    mf.register_defaults({
        "seed": -1              # not deterministic
    })
    mf.register_helpers({
        "rng_state_gpu_torch": None,
        "rng_state_cpu_torch": None,
        "rng_state_cpu_numpy": None,
        "rng_state_cpu_random": None
    })
    mf.register_event('init', set_seed)
    mf.register_event('set_seed', set_seed)
    mf.register_event('rng_save', rng_save)
    mf.register_event('rng_restore', rng_restore)
