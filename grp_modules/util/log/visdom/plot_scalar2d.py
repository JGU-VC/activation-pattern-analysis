import numpy as np


# vector over time
def plot_scalar2d(state, Z, X=None, Y=None, title="Scalar2D", **kwargs):
    if Y is None:
        Y = state["main.examples_seen"]
    if X is None:
        assert Z.dim() == 1
        for i, z in enumerate(Z):
            plot_scalar2d(state, z, i, title=title, **kwargs)
        return

    # plotname & title
    pname = "scalar2d-" + title
    kwargs["title"] = title

    # create a new 2D-Array of values
    if pname not in state["Scalar2D"]:
        vals = np.zeros((2, 2))
        vals[0, 0] = Z
        state["Scalar2D"][pname] = {
            "x": [X],
            "y": [Y],
            "z": vals
        }

    # get old 2D-Array of values
    vals = state["Scalar2D"][pname]["z"]
    xlabels = state["Scalar2D"][pname]["x"]
    ylabels = state["Scalar2D"][pname]["y"]

    # check if positions already known
    if X in xlabels:
        x_i = xlabels.index(X)
    else:
        x_i = len(xlabels)
        xlabels.append(X)
    if Y in ylabels:
        y_i = ylabels.index(Y)
    else:
        y_i = len(ylabels)
        ylabels.append(Y)

    # check if has to be resized
    if vals.shape[0] <= x_i or vals.shape[1] <= y_i:
        new_shape = np.maximum(vals.shape, np.array([x_i, y_i]) + 1)
        new_vals = np.zeros(new_shape)
        new_vals[:vals.shape[0], :vals.shape[1]] = vals
        vals = new_vals
        state["Scalar2D"][pname]["z"] = vals

    # finally insert value
    vals[x_i][y_i] = Z

    # skip update unless is full
    if vals.shape[0] - 1 != x_i:
        return

    # add legend/ticks
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "Iteration"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Layer"
    if "columnnames" not in kwargs:
        kwargs["columnnames"] = [str(label) for label in ylabels] if len(ylabels) > 1 else [ylabels[0], ""]
    if "rownames" not in kwargs:
        kwargs["rownames"] = [str(label) for label in xlabels] if len(xlabels) > 1 else [xlabels[0], ""]

    # create/update plot
    state["WINDOWS"][pname] = state["vis"].heatmap(
        vals,
        win=pname,
        opts=kwargs,
        # opts={'layoutopts': {'plotly': {'xaxis': dict(autorange=True, showgrid=False, zeroline=False, showline=False,autotick=True,ticks='',showticklabels=False)}}}
    )
