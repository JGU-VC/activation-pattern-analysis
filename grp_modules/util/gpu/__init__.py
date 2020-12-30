import torch.nn as nn
import torch
from colored import fg, attr


def send_net_to_device(state, event, net):
    del event  # unused

    # enables half precision
    if state["half"]:
        net = net.half()

    # copy net to main-device
    net = net.to(state["main_device"])

    # enables only multi gpu if multiple gpus are given
    if len(state["gpu"]) > 1:

        # issue: https://github.com/pytorch/pytorch/issues/16831
        if not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark:
            print(fg('red') + '-> [gpu] non-deterministic/benchmark-mode does not work with multiple gpu atm. Setting to non-benchmark mode.\n\tTo disable this Warning, use any seed.' + attr('reset'))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # pytorch wrapper actually does everything itself
        net = nn.DataParallel(net, device_ids=state["gpu"], dim=0)

    return net


def send_data_to_device(state, event, tensor):
    del event  # unused
    if state["half"]:
        tensor = tensor.half()
    return tensor.to(state["main_device"], non_blocking=True)


def send_loss_to_device(state, event, loss):
    del event  # unused
    return loss.to(state["main_device"], non_blocking=True)


def send_labels_to_device(state, event, labels):
    del event  # unused
    return labels.to(state["device_labels"], non_blocking=True)


def register(mf):
    mf.register_defaults({
        "half": False,
        "main_device": lambda state, event: "cuda:" + str(state["gpu"][0]) if state["gpu"][0] >= 0 else "cpu",
        "device_labels": lambda state, event: state["main_device"],
        "loss_labels": lambda state, event: state["main_device"],
    })
    mf.register_globals({
        "gpu": [0],
    })

    mf.register_event('send_net_to_device', send_net_to_device, unique=True)
    mf.register_event('send_data_to_device', send_data_to_device, unique=True)
    mf.register_event('send_labels_to_device', send_labels_to_device, unique=True)
    mf.register_event('send_loss_to_device', send_loss_to_device, unique=True)
