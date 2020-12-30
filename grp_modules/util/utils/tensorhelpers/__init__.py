
def flatten(net):
    if net.dim() == 4:
        num_filters = net.shape[1]
        return net.permute([0, 2, 3, 1]).reshape([-1, num_filters])

    return net


def unflatten(net, orig_shape):
    if len(orig_shape) == 4:
        return net.reshape([orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1]]).permute([0, 3, 1, 2])

    return net.reshape(orig_shape)


def register(mf):
    mf.register_event('flatten', flatten, unique=True)
    mf.register_event('unflatten', unflatten, unique=True)
