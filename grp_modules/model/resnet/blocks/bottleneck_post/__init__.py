from torch import nn


class BottleneckPost(nn.Module):

    def __init__(self, state, event, inplanes, planes, stride=1, downsample=None, dilation=1):
        del downsample  # unused
        self.state = state
        self.event = event
        super().__init__()

        assert self.state["option"] in [state["ShortcutOption"].B], "bottleneck blocks currently only support shortcut type B"
        if stride != 1 or inplanes != planes * state["expansion"]:
            self.shortcut_downsampled = nn.Sequential(
                event.filter_layer(inplanes, planes * state["expansion"], kernel_size=1, stride=stride, bias=False),
                self.event.normalization_layer(planes * state["expansion"]),
            )
        else:
            self.shortcut_downsampled = nn.Sequential()

        width = int(planes * (state["width_per_group"] / 64.)) * state["groups"]

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.block_seq = nn.Sequential(
            event.filter_layer(inplanes, width, kernel_size=1, stride=1, bias=False),
            event.normalization_layer(width),
            event.activation_layer(inplace=state["inplace"]),
            event.filter_layer(width, width, kernel_size=3, stride=stride, groups=state["groups"], padding=dilation, dilation=dilation, bias=False),
            event.normalization_layer(width),
            event.activation_layer(inplace=state["inplace"]),
            event.filter_layer(width, planes * state["expansion"], kernel_size=1, stride=1, bias=False),
            event.normalization_layer(planes * state["expansion"])
        )
        self.relu = event.activation_layer(inplace=state["inplace"])
        self.stride = stride

    def forward(self, x):
        out = self.block_seq(x)
        if self.state["short"]:
            out = out + self.shortcut_downsampled(x)
        return self.relu(out)


def pre_resblockchain(state, event):
    return (event.filter_layer(state["dataset.num_channels"], state["first_conv_filters"], kernel_size=state["first_conv_kernel_size"], stride=state["first_conv_stride"], padding=state["first_conv_padding"], bias=False),
            event.normalization_layer(state["first_conv_filters"]),
            event.activation_layer(inplace=state["inplace"]))


def post_resblockchain(in_planes):
    del in_planes  # unused
    return []


def register(mf):
    mf.load("..util")
    mf.set_scope("...")
    mf.register_defaults({
        "short": True,
        "option": mf.state["ShortcutOption"].B,
        "expansion": 4,
        "groups": 1,
        "width_per_group": 64,

        # first conv layer
        "first_conv_filters": int,
        "first_conv_kernel_size": int,
        "first_conv_padding": int,
        "first_conv_stride": int,
    })
    mf.register_event("resblock", BottleneckPost, unique=True)
    mf.register_event("pre_resblockchain", pre_resblockchain, unique=True)
    mf.register_event("post_resblockchain", post_resblockchain, unique=True)
