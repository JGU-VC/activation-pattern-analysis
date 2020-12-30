from torch import nn
from colored import fg, attr


def activation(state, *args, inplace=False, **kwargs):
    if inplace and not state["warning_applied"]:
        print(fg('red') + attr('bold') + "Warning:" + attr('reset') + fg(
            'red') + " PReLU cannot be used as an inplace activation." + attr('reset'))
        state["warning_applied"] = True
    return nn.PReLU(*args, init=state["init"], **kwargs)


def init_filter_layer_fallback(state, m):
    nn.init.kaiming_normal_(m.weight, a=state["init"], mode='fan_out', nonlinearity='leaky_relu')


def register(mf):
    mf.register_event('activation_layer_cls', lambda: nn.PReLU, unique=True)
    mf.register_event('activation_layer', activation, unique=True)
    mf.register_event('init_filter_layer_fallback', init_filter_layer_fallback, unique=True)
    mf.register_defaults({
        "init": 0.25,
    })
    mf.register_helpers({
        "warning_applied": False
    })
