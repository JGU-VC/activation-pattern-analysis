from miniflask import get_default_args


def register_normalization_with_defaults(mf, Normalization, overwrite=None, init_normalization=None, additional_args=None):
    if not init_normalization:
        def init_normalization(state, *args):
            return Normalization(*args, **{k: state[k] for k in state["arg_names"]})

    args = get_default_args(Normalization)
    if overwrite:
        args.update(overwrite)
    mf.register_defaults(args)
    mf.register_helpers({
        "arg_names": list(args.keys()) + [] if additional_args is None else additional_args
    })
    mf.register_event('normalization_layer', init_normalization, unique=True)
    mf.register_event('normalization_layer_cls', lambda: Normalization, unique=True)


def register(mf):
    mf.register_event('register_normalization_with_defaults', register_normalization_with_defaults, unique=True)
