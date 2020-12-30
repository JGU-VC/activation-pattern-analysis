import numpy as np


# vector over time
def plot_bar(state, Z, X=None, title="Bar", **kwargs):
    if X is None:
        assert Z.dim() == 1
        for i, z in enumerate(Z):
            plot_bar(state, z, i, title=title, **kwargs)
        return
    if isinstance(X, (float, int)):
        X = [X]

    # plotname & title
    pname = "bar-" + title
    kwargs["title"] = title

    # create a new 2D-Array of values
    if pname not in state["Bar"]:
        vals = np.zeros((1))
        vals[0] = Z
        state["Bar"][pname] = {
            "x": X,
            "z": vals,
        }

    # get old Array of values
    vals = state["Bar"][pname]["z"]
    xlabels = state["Bar"][pname]["x"]

    # check if positions already known
    if X in xlabels:
        x_i = xlabels.index(X)
    else:
        x_i = len(xlabels)
        xlabels.append(X)

    # check if has to be resized
    if vals.shape[0] <= x_i:
        new_shape = np.maximum(vals.shape, np.array([x_i]) + 1)
        new_vals = np.zeros(new_shape)
        new_vals[:vals.shape[0]] = vals
        vals = new_vals
        state["Bar"][pname]["z"] = vals

    # finally insert value
    vals[x_i] = Z

    # skip update unless is full
    if vals.shape[0] - 1 != x_i:
        return

    # add legend/ticks
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "Iteration"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Layer"
    if "rownames" not in kwargs:
        kwargs["rownames"] = [str(label) for label in xlabels] if len(xlabels) > 1 else [xlabels[0], ""]

    # create/update plot
    state["WINDOWS"][pname] = state["vis"].bar(
        vals,
        win=pname,
        opts=kwargs,
    )
