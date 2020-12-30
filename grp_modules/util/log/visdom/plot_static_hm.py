
# static heatmap
def plot_static_hm(state, Z, X=None, Y=None, title="Heatmap", **kwargs):
    del X, Y  # unused
    assert Z.dim() == 2
    # plotname & title
    pname = "scalar3d-" + title
    kwargs["title"] = title

    # add legend/ticks
    if "xlabel" not in kwargs:
        kwargs["xlabel"] = "X"
    if "ylabel" not in kwargs:
        kwargs["ylabel"] = "Y"

    # create/update plot
    state["WINDOWS"][pname] = state["vis"].heatmap(
        Z.T,
        win=pname,
        opts=kwargs,
    )
