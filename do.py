#!/bin/python
import sys
import os
sys.path.insert(0, os.path.join("util","miniflask","src"))
import miniflask

# check if user tries to profile this script
import torch
if "torch.utils.bottleneck" in sys.modules:
    print("BOTTLENECK MODE")
    torch.multiprocessing.set_start_method('spawn', force="True")
torch.backends.cudnn.benchmark=True

# init miniflask
mf = miniflask.init(
    # module_dirs=["./grp_modules", "./entropy_modules"]
    module_dirs={
        "grp": "./grp_modules",
        "actpat": "./actpat_modules"
    }
    #,debug=True
)

# load autoload modules for this project
mf.register_default_module("cpuloader", required_event="dataloader", overwrite_globals={
    "num_workers": 8
})
mf.run(["log", "settings", "stop-at-divergence"])
