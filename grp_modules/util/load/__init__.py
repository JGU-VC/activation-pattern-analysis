import os
import torch
from miniflask import print_info as print  # pylint: disable=W0622


def load(state, *args, net=None, **kwargs):
    if net is None:
        net = state["main.net"]

    ckpt_file = os.path.join(state["dir"], state["filename"])
    if os.path.isfile(ckpt_file):

        # load checkpoint
        print("... loading ckpt '{}'".format(ckpt_file))
        checkpoint = torch.load(ckpt_file)

        # assert that the model matches
        # assert type(net) == checkpoint["model_type"], "Model does not match"

        # save checkpoint metadata
        state["main.start_epoch"] = checkpoint['epoch']
        state["best_acc"] = checkpoint['best_acc']

        # load model weights
        net.load_state_dict(checkpoint['state_dict'])
        print("... loaded ckpt '{}' (saved epoch {})".format(ckpt_file, checkpoint['epoch']))
    else:
        print("... no ckpt found at '{}'".format(ckpt_file))

    return args, kwargs


def register(mf):
    mf.register_defaults({
        "filename": lambda state, event: state["log.tag"] + ".ckpt" if "log.tag" in state else "model.ckpt",
        "dir": "./ckpts",
    })
    mf.register_event('before_send_net_to_device', load)
    mf.register_event('load_ckpt', load, unique=True)
