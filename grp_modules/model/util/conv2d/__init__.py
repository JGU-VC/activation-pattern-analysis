from torch.nn import Conv2d
from torch.nn.init import kaiming_normal_


def init_filter_layer_fallback(m):
    kaiming_normal_(m.weight)


def filter_layer(event, in_channels, out_channels, kernel_size, initializer=None, **kwargs):
    layer = Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    # explanation of this filter_layer initialization scheme:
    # (Reason for this design is that we want to reduce unneeded gpu-RNG calls)
    # - init_filter_layer is an non-unique event that can be implemented by several modules
    #   by default, it calls the init_filter_layer_fallback
    # - init_filter_layer_fallback can be specialized as well by any module
    #   by default it calls kaiming_normal_
    # - the used case is:
    #   * the activation (or layers that follow the filter layer) function defines the init_filetr_layer_fallback
    #   * any other initialization method wraps that event and should fall back back to the default if it is not affecting the given module
    if initializer is None:
        def default_init_filter(m):
            event.optional.init_filter_layer_fallback(m, altfn=init_filter_layer_fallback)
        event.optional.init_filter_layer(layer, altfn=default_init_filter)
    else:
        initializer(layer)
    return layer


def register(mf):
    mf.register_event("filter_layer", filter_layer, unique=True)
    mf.register_event("filter_layer_cls", lambda: Conv2d, unique=True)
