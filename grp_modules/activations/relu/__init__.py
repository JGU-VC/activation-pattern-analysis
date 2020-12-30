from torch import nn


def activation(*args, **kwargs):
    return nn.ReLU(*args, **kwargs)


def init_filter_layer_fallback(m, altfn=None):
    del altfn
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def register(mf):
    mf.register_event('activation_layer_cls', lambda: nn.ReLU, unique=True)
    mf.register_event('activation_layer', activation, unique=True)
    mf.register_event('init_filter_layer_fallback', init_filter_layer_fallback, unique=True)
