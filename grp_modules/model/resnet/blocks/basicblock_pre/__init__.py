from torch import nn
import torch.nn.functional as F


class BasicBlockPre(nn.Module):

    def __init__(self, state, event, in_planes, planes, *args, stride=1, **kwargs):
        del args, kwargs  # unused
        self.event = event
        self.state = state
        super().__init__()

        self.block_seq = nn.Sequential(
            event.normalization_layer(in_planes),
            event.activation_layer(inplace=state["inplace"]),
            event.filter_layer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            event.normalization_layer(planes),
            event.activation_layer(inplace=state["inplace"]),
            event.filter_layer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        )

        assert self.state["option"] in [state["ShortcutOption"].A, state["ShortcutOption"].B], "basic blocks currently only support shortcut types A and B"
        if stride != 1 or in_planes != planes:

            # In the original ResNet paper, Cifar10 uses option A, ImageNet uses option B.
            if self.state["option"] == state["ShortcutOption"].A:
                self.shortcut = state["LambdaLayer"](lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0))
            elif self.state["option"] == state["ShortcutOption"].B:
                self.shortcut = nn.Sequential(
                    event.filter_layer(in_planes, state["expansion"] * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(state["expansion"] * planes)
                )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.block_seq(x)
        if self.state["short"]:
            out = out + self.shortcut(x)
        return out


def pre_resblockchain(event, state):
    return (event.filter_layer(state["dataset.num_channels"], state["first_conv_filters"], kernel_size=state["first_conv_kernel_size"], stride=state["first_conv_stride"], padding=state["first_conv_padding"], bias=False), )


def post_resblockchain(state, event, in_planes):
    return (event.normalization_layer(in_planes),
            event.activation_layer(inplace=state["inplace"]))


def register(mf):
    mf.load("..util")
    mf.set_scope("...")
    mf.register_defaults({
        "short": True,
        "option": mf.state["ShortcutOption"].A,
        "expansion": 1,

        # first conv layer
        "first_conv_filters": int,
        "first_conv_kernel_size": int,
        "first_conv_padding": int,
        "first_conv_stride": int,
    })
    mf.register_event("resblock", BasicBlockPre, unique=True)
    mf.register_event("pre_resblockchain", pre_resblockchain, unique=True)
    mf.register_event("post_resblockchain", post_resblockchain, unique=True)
