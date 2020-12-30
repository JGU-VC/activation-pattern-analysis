import os
import torch


def save(state, event, net=None):
    print("save")
    print("save")
    print("save")
    if net is None:
        net = state["main.net"]
    if not os.path.exists(state["dir"]):
        os.makedirs(state["dir"])

    ckpt_file = os.path.join(state["dir"], state["filename"])
    net = state["main.net"]

    epoch = state["main.current_epoch"]
    current_acc = state["val_accuracy"] if hasattr(event, 'validate') and "val_accuracy" in state else state["last_accuracy"]
    best_acc = state["best_acc"] if "best_acc" in state else 0

    # remember best prec@1 and save checkpoint
    is_best = current_acc > best_acc  # todo: remove statement, unused
    del is_best  # unused
    best_acc = max(current_acc, best_acc)
    state["best_acc"] = best_acc

    if epoch >= 0 and state["every_epoch"] > 0 and epoch % state["every_epoch"] == 0:
        torch.save({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
        }, ckpt_file)


def save_now(state, net=None):
    print("save now")
    print("save now")
    print("save now")
    if net is None:
        net = state["main.net"]
    if not os.path.exists(state["dir"]):
        os.makedirs(state["dir"])

    ckpt_file = os.path.join(state["dir"], state["filename"])
    epoch = state["main.current_epoch"]
    best_acc = state["best_acc"]
    current_acc = state["val_accuracy"]
    torch.save({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'final_acc': current_acc,
    }, ckpt_file)


def register(mf):
    mf.register_defaults({
        "filename": lambda state, event: state["log.tag"] + ".ckpt" if "log.tag" in state else "model.ckpt",
        "every_epoch": lambda state, event: int(state["main.epochs"] * 0.1),
        "dir": "./ckpts",
    })
    mf.register_event('after_epoch', save)
    mf.register_event('after_training', save_now)
    mf.register_event('save_ckpt', save)
