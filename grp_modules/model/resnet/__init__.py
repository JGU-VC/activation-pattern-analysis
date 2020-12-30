from enum import Enum
import torch.nn as nn


class PlaneIncreaseStrategy(Enum):
    DEFAULT = 0
    ADD = 1


class ResNet(nn.Module):

    def __init__(self, state, event):
        assert len(state["replace_stride_with_dilation"]) + 1 == len(state["strides"]) and len(state["strides"]) == len(state["num_blocks"]), "len(replace_stride_with_dilation)+1, len(strides) should be same size as len(num_blocks)"
        assert len(state["num_planes"]) == (len(state["num_blocks"]) if state["plane_increase_strategy"] == state["PlaneIncreaseStrategy"].DEFAULT else 2)
        super().__init__()
        block = event.resblock
        self.state = state
        self.event = event
        self.in_planes = state["first_conv_filters"]
        self.float_planes = state["num_planes"][0]
        self.dilation = 1

        # first conv ensures right number of channels
        self.pre_resblockchain = nn.Sequential(
            *event.pre_resblockchain(),
            *([event.activation_layer(inplace=False)] if state["first_conv_activation"] else []),  # inplace=True causes a problem here, not sure why
            *([nn.MaxPool2d(kernel_size=3, stride=state["strides"][0], padding=1)] if state["first_conv_max_pool"] else [])
        )

        # residual blocks
        # note: first residual blocks may differ if first_conv applies max pooling
        self.resblocks = []
        self.blocks = nn.ModuleList([])
        for i, (num_planes, num_blocks, stride, dilate) in enumerate(zip((state["num_planes"] if state["plane_increase_strategy"] == state["PlaneIncreaseStrategy"].DEFAULT else [None] * len(state["num_blocks"])), state["num_blocks"], state["strides"], [False] + state["replace_stride_with_dilation"])):
            if i == 0 and state["first_conv_max_pool"]:
                stride = 1
            self.blocks.append(self._make_layer(state, block, num_planes, num_blocks, stride=stride, dilate=dilate))

        # shortcut-path may not have seen any batchnorm
        self.post_resblockchain = nn.Sequential(
            *event.post_resblockchain(self.in_planes)
        )

        # global average pooling & fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(state["num_planes"][-1] * state["expansion"], state["dataset.num_classes"])

        # full sequence
        self.full_sequence = nn.Sequential(
            self.pre_resblockchain,
            *self.blocks,
            self.post_resblockchain,
            self.avgpool,
            nn.Flatten(),
            self.fc,
        )

        # optional: apply weight initializer
        event.optional.init_net_finished(self)

    # residual blocks
    def _make_layer(self, state, block, planes, num_blocks, stride=1, dilate=False):
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # note: the first block defined differs from the rest of the block-chain
        layers = []
        for i in range(num_blocks):
            self.float_planes = {
                state["PlaneIncreaseStrategy"].DEFAULT: planes,
                state["PlaneIncreaseStrategy"].ADD: self.float_planes + (state["num_planes"][1] - state["num_planes"][0]) / sum(state["num_blocks"])
            }[state["plane_increase_strategy"]]
            out_planes = int(round(self.float_planes))
            block_ = block(self.in_planes, out_planes, stride=stride if i == 0 else 1, dilation=previous_dilation if i == 0 else self.dilation)
            self.in_planes = out_planes * state["expansion"]
            layers.append(block_)
            self.resblocks.append(block_)

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.full_sequence(x)


def register(mf):

    # defaults of resnet20 / cifar-variant
    mf.register_defaults({
        # base settings
        "num_blocks": [int],
        "num_planes": [int],
        "strides": [int],

        # each element in the tuple indicates if we should replace
        # the 2x2 stride with a dilated convolution instead
        "replace_stride_with_dilation": [bool],

        # after pre_resblockchain (before first res-block), apply activation and/or pooling?
        "first_conv_activation": bool,
        "first_conv_max_pool": bool,

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        "zero_init_residual": False,
        "inplace": True,
        "plane_increase_strategy": PlaneIncreaseStrategy,
    })
    mf.register_helpers({
        "PlaneIncreaseStrategy": PlaneIncreaseStrategy,
    })
    mf.register_event("init_net", ResNet, unique=True)
