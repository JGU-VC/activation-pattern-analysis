from torch.nn import Identity


def normalization_layer(state, event, num_channels):
    del state, event, num_channels  # unused
    return Identity()


def register(mf):
    mf.register_event('normalization_layer', normalization_layer, unique=True)
