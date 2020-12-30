from torch.nn import Linear


def classifier_layer(state, in_channels, out_channels=None, bias=None, **kwargs):
    del kwargs

    if out_channels is None:
        out_channels = state["dataset.num_classes"]
    if bias is None:
        bias = state["bias"]

    return Linear(in_channels, out_channels, bias=bias)


def register(mf):
    mf.register_defaults({
        "bias": True
    })
    mf.register_event("classifier_layer", classifier_layer, unique=True)
    mf.register_event("classifier_layer_cls", lambda: Linear, unique=True)
