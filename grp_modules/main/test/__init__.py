from tqdm import tqdm
import torch


def test(state, event, *args, testloader=None, net=None, evalmode=True, plot=True, **kwargs):
    # breakpoint()
    del args, kwargs  # unused
    if not testloader:
        testloader = event.dataloader("test")
    if not net:

        if "main.net" in state:
            net = state["main.net"]
        else:

            # get network
            net = state["net"] = event.init_net()

            # send to device
            net = event.send_net_to_device(net)

            # get criterion
            criterion = event.init_loss()
            criterion = event.send_loss_to_device(criterion)

    # optional events (typically optimizer, learning rate scheduler, etc.)
    event.optional.before_testing()

    # testing loop
    event._mf.print_heading("Testing Loop ...")
    state["num_batches"] = len(testloader)

    if evalmode:
        net.eval()
    acc_1 = event.Welford()
    acc_5 = event.Welford()
    with torch.no_grad():
        if "tqdm_batch" not in state:
            state["tqdm_batch"] = tqdm(total=len(testloader), position=0, desc="Test", dynamic_ncols=False)
        for state["current_batch"], data in enumerate(testloader):
            _inputs = event.send_data_to_device(data[0])
            _labels = event.send_labels_to_device(data[1])
            state["main.labels"] = _labels
            output = net(_inputs)
            state["examples_seen"] += len(_inputs)
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

    event.optional.after_testing()

    state["test_accuracy"] = acc_1.mean
    net.train()

    if plot:
        event.optional.plot_scalar(acc_1.mean, 0, title="test_acc_1")
        event.optional.plot_scalar(acc_5.mean, 0, title="test_acc_5")

    event.optional.reset_dataloader("test")

    # pre-delete tqdm object (garbage collection fires execiption due to bug in in tqdm related to `dynamic_ncols=False`)
    # if "tqdm_batch" in state:
    #     del state["tqdm_batch"]


def register(mf):
    mf.set_scope("..")
    mf.register_helpers({
        "current_epoch": 0,
        "current_batch": 0,
        "num_batches": 0,
        "examples_seen": 0
    })
    mf.load('Welford')
    mf.register_event('main', test)
    mf.register_event('test', test)
