# PyramidNet Architecture for CIFAR10
# with the "basicblock_pyramid" block, the resulting architecture is PyramidNet110 (set as default)
# with the "bottleneck_pyramid" block, the resulting architecture is PyramidNet164

def register(mf):
    mf.register_default_module("basicblock_pyramid", required_event="resblock")
    mf.load("grp.model.resnet.cifar_variants.pyramidnet_defaults")
    mf.set_scope("grp.model.resnet")
    mf.overwrite_defaults({
        "num_blocks": 3 * [3],  # first_layer + (3 blocks * 18 reslayers * 2 convs) + final_layer = 110 layers
    }, scope="grp.model.resnet")
