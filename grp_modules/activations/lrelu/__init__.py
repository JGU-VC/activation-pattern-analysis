from torch import nn


def activation(state, *args, **kwargs):
    return nn.LeakyReLU(*args, negative_slope=state["alpha"], **kwargs)


def init_filter_layer_fallback(state, m):
    if state["mode"] == "kaiming":
        nn.init.kaiming_normal_(m.weight, a=state["alpha"], mode='fan_out', nonlinearity='leaky_relu')
    elif state["mode"] == "orthogonal":
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("leaky_relu", state["alpha"]))


def register(mf):
    mf.register_event('activation_layer_cls', lambda: nn.LeakyReLU, unique=True)
    mf.register_event('activation_layer', activation, unique=True)
    mf.register_event('init_filter_layer_fallback', init_filter_layer_fallback, unique=True)
    mf.register_defaults({
        "mode": "kaiming",  # kaiming or orthogonal
        "alpha": 0.01,
    })

    mf.register_helpers({
        "relus": nn.ModuleList(),
        "warning_applied": False,
    }, parsefn=False, scope="")
