# define optimization techniques
import torch
from miniflask import outervar, like, get_default_args


def eval_loss(event, net, inputs, labels, criterion, call_after_fwd_pass=True):

    # get result & loss
    outputs = net(inputs)
    if call_after_fwd_pass:
        event.optional.after_fwd_pass(net)
    current_loss = torch.mean(criterion(outputs, labels))
    regularizer = sum(event.optional.regularizer(net))

    # collect all losses
    if hasattr(event, 'mix_total_loss'):
        return event.mix_total_loss(current_loss, regularizer), outputs, current_loss

    return current_loss + regularizer, outputs, current_loss


def optimize(state, event, inputs, labels, net, criterion):
    optimizer = state["optimizer"]

    # init step
    optimizer.zero_grad()

    total_loss, outputs, current_loss = eval_loss(event, net, inputs, labels, criterion)

    # calculate gradients (with possibility to hook)
    event.optional.backward(total_loss, altfn=lambda x: x.backward())

    # apply optimizer
    optimizer.step(lambda: eval_loss(event, net, inputs, labels, criterion, False)[0] if state["step_with_closure"] else None)

    # get acc
    current_acc = get_acc(outputs, labels, net)
    state["last_accuracy"] = current_acc

    # event after every step
    plot_step(state, event, current_loss, current_acc)

    return current_loss, current_acc


def get_acc(outputs, labels, net):
    del net  # unused
    with torch.no_grad():
        try:
            _, predicted = outputs.max(dim=1)
        # todo: find out which exception(s) actually can occur.
        except Exception as e:  # noqa: E722
            raise RuntimeError("Optimize must be called before get_acc") from e
        correct = (predicted == labels).sum().double()
        total = torch.tensor(labels.size(0)).double().to(correct.device)
        acc = correct / total
        return acc


def plot_step(state, event, current_loss, current_acc):
    if event.optional.plot_every(state["plot.steps"]):
        if len(state["optimizer"].param_groups) == 1:
            event.optional.plot_scalar(state["optimizer"].param_groups[0]['lr'], title="learning rate")
        else:
            for group_id, parameters in enumerate(state["optimizer"].param_groups):
                event.optional.plot_scalar2d(parameters['lr'], group_id, title="learning rate per group")
            mean_lr = torch.mean(torch.tensor([parameters['lr'] for parameters in state["optimizer"].param_groups]))
            event.optional.plot_scalar(mean_lr, title="learning rate")
        event.optional.plot_scalar(current_loss.item(), title="loss")
        event.optional.plot_scalar(current_acc.item(), title="accuracy")


def call_init_optimizer(state, event, net=outervar):
    state["optimizer"] = event.init_optimizer(net, net.parameters())


def get_optimizer(state, event):
    del event  # unused
    if state["optimizer"] is None:
        raise ValueError("Optimizer not set yet. Maybe you used the wrong order in your module loading?")
    return state["optimizer"]


def register_optimizer_with_defaults(event, mf, Optimizer, with_closure=False, overwrite=None, init_optimizer=None):
    if not init_optimizer:
        def init_optimizer(state, net, parameters=None):
            if not parameters:
                parameters = net.parameters()

            def default_parameter_group(net, parameters):
                del net
                return parameters

            parameters = event.optional.make_optimizer_parameter_groups(net, parameters, altfn=default_parameter_group)
            return Optimizer(parameters, **{k: state[k] for k in state["arg_names"]})

    args = get_default_args(Optimizer)
    if overwrite:
        args.update(overwrite)
    mf.register_defaults(args)
    mf.register_helpers({
        "arg_names": args.keys()
    })
    if with_closure:
        mf.overwrite_defaults({
            "step_with_closure": True
        })
    mf.register_event('init_optimizer', init_optimizer, unique=True)


def register(mf):
    mf.register_default_module("sgd", required_event="init_optimizer")
    mf.register_defaults({
        "plot.steps": like("log.steps.scalar", alt=0),
        "step_with_closure": False,
    })
    mf.register_helpers({
        "optimizer": None,
    })
    mf.register_event('step', optimize, unique=True)
    mf.register_event('before_training', call_init_optimizer)
    mf.register_event('get_optimizer', get_optimizer, unique=True)
    mf.register_event('register_optimizer_with_defaults', register_optimizer_with_defaults, unique=True)
