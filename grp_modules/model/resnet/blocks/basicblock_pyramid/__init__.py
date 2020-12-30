"""
Basic building block for the PyramidNet architecture.

Paper:
https://arxiv.org/abs/1610.02915

This code is mostly based on the pytorch implementation by Dongyoon Han, available on Github:
https://github.com/dyhan0920/PyramidNet-PyTorch
"""

from torch import nn
import torch.nn.functional as F


class BasicBlockPyramid(nn.Module):

    def __init__(self, state, event, in_planes, planes, *args, stride=1, **kwargs):
        del args, kwargs  # unused
        self.event = event
        self.state = state

        super().__init__()

        out_planes = planes * state["expansion"]

        self.block_seq = nn.Sequential(
            event.normalization_layer(in_planes),
            # first relu from preresnet is removed in pyramidnet
            event.filter_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            event.normalization_layer(planes),
            event.activation_layer(inplace=state["inplace"]),
            event.filter_layer(planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            event.normalization_layer(out_planes),  # normalization layer added at the end of the block
        )

        # build shortcut
        assert self.state["option"] in [state["ShortcutOption"].A, state["ShortcutOption"].B], "pyramid blocks currently only support shortcut types A and B"
        pool, stride = (1, stride) if self.state['downsample'] == 'strides' else (stride, 1)
        inc_channels = out_planes - in_planes
        self.shortcut = nn.Sequential(
            *([{'avgpool': nn.AvgPool2d}[self.state['downsample']]((pool,) * 2, stride=(pool,) * 2, ceil_mode=True)]
              if pool != 1 else []),
            *([event.filter_layer(in_planes, out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_planes)]
              if inc_channels and state["option"] == state["ShortcutOption"].B else
              [*([state["LambdaLayer"](lambda x: x[..., ::stride, ::stride])] if stride != 1 else []),
               *([state["LambdaLayer"](lambda x: F.pad(x, [0, 0, 0, 0, 0, inc_channels]))] if inc_channels else [])]))

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**0.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block_seq(x)
        if self.state["short"]:
            out = out + self.shortcut(x)
        return out


def pre_resblockchain(state, event):
    return (event.filter_layer(state["dataset.num_channels"], state["first_conv_filters"], kernel_size=state["first_conv_kernel_size"], stride=state["first_conv_stride"], padding=state["first_conv_padding"], bias=False),
            event.normalization_layer(state["first_conv_filters"]))


def post_resblockchain(state, event, in_planes):
    return (event.normalization_layer(in_planes),
            event.activation_layer(inplace=state["inplace"]))


def register(mf):
    mf.load("..util")
    mf.set_scope("...")
    mf.register_defaults({
        "short": True,
        "option": mf.state["ShortcutOption"].A,  # original paper recommends and uses only state["ShortcutOption"].A-type
        "downsample": 'avgpool',
        "expansion": 1,

        # first conv layer
        "first_conv_filters": int,
        "first_conv_kernel_size": int,
        "first_conv_padding": int,
        "first_conv_stride": int,
    })
    mf.register_event("resblock", BasicBlockPyramid, unique=True)
    mf.register_event("pre_resblockchain", pre_resblockchain, unique=True)
    mf.register_event("post_resblockchain", post_resblockchain, unique=True)
