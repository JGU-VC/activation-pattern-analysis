def plot_scatter(state, X, Y=None, title="Scatter", update=False, **kwargs):
    '''X is a Nx2 or Nx3 Tensor that specifies the locations of N points in the scatter plot.
    Y is a N Tensor that contains discrete labels.
    See https://github.com/facebookresearch/visdom#visscatter for all possible parameters.'''

    # plotname & title
    pname = "scalar-" + title
    kwargs["title"] = title

    # helper object
    append = pname in state["WINDOWS"]

    obj = state["WINDOWS"][pname] = {}
    del obj  # unused

    # shall append ?
    update_kwarg = dict(update="append") if append and update else {}
    # create/update plot
    state["vis"].scatter(
        Y=Y,
        X=X,
        win=pname,
        opts=kwargs,
        **update_kwarg
    )
