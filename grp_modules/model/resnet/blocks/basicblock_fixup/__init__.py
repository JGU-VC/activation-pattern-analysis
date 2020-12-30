import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class ScaleLayer(nn.Module):
    def __init__(self, channels, skip_dims=2):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels, *[1] * skip_dims))

    def forward(self, net):
        return net * self.scale

    def extra_repr(self):
        return f'shape={self.scale.shape}'


class BiasLayer(nn.Module):
    def __init__(self, channels, skip_dims=2):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels, *[1] * skip_dims))

    def forward(self, net):
        return net + self.bias

    def extra_repr(self):
        return f'shape={self.bias.shape}'


class BasicBlockFixup(nn.Module):

    def __init__(self, state, event, in_planes, planes, *args, stride=1, **kwargs):
        del args, kwargs  # unused
        super().__init__()
        self.event = event
        self.state = state

        self.bias1 = BiasLayer(1)

        def conv1_init(layer):
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(2 / (layer.weight.shape[0] * np.prod(layer.weight.shape[2:]))) * sum(state['num_blocks']) ** (-0.5))
        self.conv1 = event.filter_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, initializer=conv1_init)

        def conv2_init(layer):
            nn.init.constant_(layer.weight, 0)
        self.conv2 = event.filter_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, initializer=conv2_init)

        self.block_seq = nn.Sequential(
            self.bias1,
            self.conv1,
            BiasLayer(1),
            event.activation_layer(inplace=state["inplace"]),
            BiasLayer(1),
            self.conv2,
            ScaleLayer(1),
            BiasLayer(1),
        )
        self.relu = event.activation_layer(inplace=state["inplace"])

        if stride != 1 or in_planes != planes:

            # Cifar-10 ResNet paper uses option A.
            if self.state["option"] == state["ShortcutOption"].A:
                self.shortcut = state["LambdaLayer"](lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0))
            elif self.state["option"] == state["ShortcutOption"].B:
                self.shortcut = nn.Sequential(
                    event.filter_layer(in_planes, state["expansion"] * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(state["expansion"] * planes)
                )
            self.shortcut = nn.Sequential(self.bias1, self.shortcut)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.block_seq(x)
        if self.state["short"]:
            out = out + self.shortcut(x)
        return self.relu(out)


def pre_resblockchain(state, event):
    return (event.filter_layer(state["dataset.num_channels"], state["first_conv_filters"], kernel_size=state["first_conv_kernel_size"], stride=state["first_conv_stride"], padding=state["first_conv_padding"], bias=False),
            BiasLayer(1),
            event.activation_layer(inplace=state["inplace"]))


def post_resblockchain(in_planes):
    del in_planes
    return (BiasLayer(1), )


def make_optimizer_parameter_groups(state, net, parameters):
    del parameters
    print('\nUsing Fixup learning rate reduction by 1/10th for biases and scaling factors')
    # override parameters with 1/10th lr for bias an scale
    parameters_bias = [p[1] for p in net.named_parameters() if 'bias' in p[0]]
    parameters_scale = [p[1] for p in net.named_parameters() if 'scale' in p[0]]
    parameters_others = [p[1] for p in net.named_parameters() if not ('bias' in p[0] or 'scale' in p[0])]
    return [
        {'params': parameters_bias, 'lr': state['lr'] / 10.},
        {'params': parameters_scale, 'lr': state['lr'] / 10.},
        {'params': parameters_others}
    ]


def register(mf):
    mf.load("..util")
    mf.set_scope("...")
    mf.load("identity-classifier")
    mf.register_defaults({
        "short": True,
        "concat": False,
        "option": mf.state["ShortcutOption"].A,
        "expansion": 1,

        # first conv filter
        "first_conv_filters": int,
        "first_conv_kernel_size": int,
        "first_conv_padding": int,
        "first_conv_stride": int,
    })
    mf.register_event("resblock", BasicBlockFixup, unique=True)
    mf.register_event("pre_resblockchain", pre_resblockchain, unique=True)
    mf.register_event("post_resblockchain", post_resblockchain, unique=True)
    mf.register_event("make_optimizer_parameter_groups", make_optimizer_parameter_groups, unique=True)
