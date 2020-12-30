from tqdm import tqdm
import torch
from miniflask import outervar


def validate(state, event, *args, valloader=None, net=outervar, evalmode=True, plot=True, dummy=False, **kwargs):
    del args, kwargs  # unused
    if not valloader:
        valloader = event.dataloader("val")

    if valloader is not None:
        if evalmode:
            net.eval()
        acc_1 = event.Welford()
        acc_5 = event.Welford()
        with torch.no_grad():
            if "tqdm_batch" not in state:
                state["tqdm_batch"] = tqdm(total=len(valloader), position=3, desc="Validation", dynamic_ncols=False)
            for _, data in enumerate(valloader):
                if dummy:
                    continue
                _inputs = event.send_data_to_device(data[0])
                _labels = event.send_labels_to_device(data[1])
                state["main.labels"] = _labels
                output = net(_inputs)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                _labels = _labels.view(_labels.size(0), -1).expand_as(pred)
                correct = pred.eq(_labels).float()

                # compute top-1/top-5
                correct_5 = correct[:, :5].sum(1).cpu().numpy()
                correct_1 = correct[:, :1].sum(1).cpu().numpy()

                [acc_1(c) for c in correct_1]  # pylint: disable=expression-not-assigned
                [acc_5(c) for c in correct_5]  # pylint: disable=expression-not-assigned

                state["tqdm_batch"].update(1)
            state["tqdm_batch"].reset()
            state["tqdm_batch"].clear()

        state["val_accuracy"] = acc_1.mean
        net.train()

        if plot:
            event.optional.plot_scalar(acc_1.mean, title="validation_acc_1")
            event.optional.plot_scalar(acc_5.mean, title="validation_acc_5")

    event.optional.reset_dataloader("val")


def register(mf):
    mf.load('Welford')
    mf.register_event('after_epoch', validate)
    # mf.register_event('after_training', validate)
    mf.register_event('validate', validate)
