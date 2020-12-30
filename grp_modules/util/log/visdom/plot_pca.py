import torch


def plot_pca(state, event, data, bias, plot_id):
    k = 2
    X = event.flatten(data)
    U, _, _ = torch.svd(torch.t(X))
    C = torch.mm(X, U[:, :k])
    b = torch.mm(bias[None].detach(), U[:, :k])
    labels = (C[:, 0] < b[:, 0]).type(torch.int) * 2 + (C[:, 1] < b[:, 1]).type(torch.int) + 1

    # add bias as a point
    C = torch.cat((C, b))
    labels = torch.cat((labels, torch.tensor([5], device=labels.device, dtype=labels.dtype)))

    size = 10
    state["vis"].scatter(C, labels.cpu().numpy(), win="pca_layer" + str(plot_id), opts={"title": "pca_" + str(plot_id), "markersize": size})
