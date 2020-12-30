from tqdm import tqdm


def main(state, event):

    # load dataset
    state["trainloader"] = event.dataloader("train")
    state["num_batches"] = len(state["trainloader"])

    # get network
    net = state["net"] = event.init_net()

    # send to device
    net = event.send_net_to_device(net)

    # get criterion
    criterion = event.init_loss()
    criterion = event.send_loss_to_device(criterion)

    # optional events (typically optimizer, learning rate scheduler, aso.)
    event.optional.before_training()

    # train loop
    event._mf.print_heading("Training Loop ...")
    net.train()
    tqdm_epoch = tqdm(total=state["epochs"], position=1, desc="Epoch", initial=state["start_epoch"], dynamic_ncols=False)
    tqdm_batch = tqdm(total=len(state["trainloader"]), position=2, desc="Batches", dynamic_ncols=False)

    for state["current_epoch"] in range(state["start_epoch"], state["epochs"]):
        event.optional.before_epoch()

        for state["current_batch"], data in enumerate(state["trainloader"]):

            # get the inputs; data is a list of [inputs, labels]
            inputs = event.send_data_to_device(data[0])
            labels = event.send_labels_to_device(data[1])
            state["labels"] = labels

            # step
            event.step(inputs, labels, net, criterion)
            state.all["step"] += 1

            state["examples_seen"] += len(inputs)
            tqdm_batch.update(1)

        tqdm_batch.reset()
        tqdm_epoch.update(1)

        # event for every epoch (eg. validate, scheduler)
        event.optional.after_epoch()

    # event after training (eg. test, saving)
    event.optional.validate()
    event.optional.after_training()


def register(mf):
    mf.register_defaults({
        "epochs": 5,
    })
    mf.register_helpers({
        "start_epoch": 0,
        "examples_seen": 0,
        # "plot.last_plot": -1
    }, scope="..")
    mf.register_helpers({
        "step": 0,
    }, scope="")
    mf.register_event('main', main)
