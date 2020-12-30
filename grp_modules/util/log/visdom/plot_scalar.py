import numpy as np


def plot_scalar(state, Y, X=None, title="Scalar", **kwargs):
    if X is None:
        X = state["main.examples_seen"]

    # plotname & title
    pname = "scalar-" + title
    kwargs["title"] = title

    # assure right format
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y).reshape([-1])
    if not isinstance(X, np.ndarray):
        X = np.array(X).reshape([-1])

    # helper object
    append = pname in state["WINDOWS"] and len(X) == 1
    state["WINDOWS"][pname] = {}

    # shall append ?
    update_kwarg = dict(update="append") if append else {}

    # create/update plot
    state["vis"].line(
        Y=Y,
        X=X,
        win=pname,
        opts=kwargs,
        **update_kwarg
    )
