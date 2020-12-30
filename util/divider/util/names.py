import numpy as np

networks = {
    "convnet20": "ConvNet-20",
    "resnet20": "ResNet-20",
    "pyramidnet20": "PyramidNet-20",
    "pyramid20": "PyramidNet-20",
    "resnet20,basicblock_pre": "ResNet-20",
    "resnet20,basicblockpre": "ResNet-20",
    "toyconv": "ToyNet-20",
    "toynet20": "ToyNet-20",
}

datasets = {
    "c10": "cifar10",
    "tiny": "tinyimagenet"
}


# train_acc = lambda d: d["train_acc"]
test_acc = lambda d: d["test_acc"]
last_train_H = lambda d: d["train_H"]
last_test_H = lambda d: d["test_H"]
# train_H_plot = lambda d: d["scalar2d-["+d["mode_data"]+"] % max Entropy"]
learning_rate_per_step = lambda d: d["scalar-learning rate"]
train_H_plot = lambda d: d["scalar2d-["+d["mode_data"]+"] Entropy"]
train_maxH_plot = lambda d: d["scalar2d-["+d["mode_data"]+"] % max Entropy"]
train_H_mean_per_step = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"] % max Entropy"]["z"]).T[-1].mean()
test_H_mean = lambda d: d["test_H"].mean()
train_H_mean = lambda d: d["train_H"].mean()
test_H_max = lambda d: d["test_H"].max()
train_H_max = lambda d: d["train_H"].max()
test_H_min = lambda d: d["test_H"].min()
train_H_min = lambda d: d["train_H"].min()
test_H_last = lambda d: d["test_H"][-1]
train_H_last = lambda d: d["train_H"][-1]
train_H_correlation = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"] % max Entropy"]["z"])
test_H_upper_sum = lambda d: d["test_H"][len(d["test_H"])//2:].sum()
train_H_upper_sum = lambda d: d["train_H"][len(d["test_H"])//2:].sum()
test_H_lower_sum = lambda d: d["test_H"][:len(d["test_H"])//2].sum()
train_H_lower_sum = lambda d: d["train_H"][:len(d["test_H"])//2].sum()
Δ_acc = lambda d: (d["train_acc"] - d["test_acc"])#/d["train_acc"]
train_step = lambda d: int(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["x"][-1])
train_step_over_time = lambda d: [int(f) for f in d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["x"]]
JIgT_per_step = lambda d: d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]
# Jaccard2last_mean = lambda d: np.median(np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1],0)
Jaccard2last_max = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1].max()
Jaccard2last_min = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1].min()
Jaccard2last_mean = lambda d: np.mean(np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1])
Jaccard2last_mean_thresh = lambda d: np.mean(np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1])
Jaccard2last_median = lambda d: np.median(np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).T[-1])
train_H_over_time = lambda d: np.mean(np.array(d["scalar2d-["+d["mode_data"]+"] % max Entropy"]["z"]), 0)
train_acc_over_time = lambda d: d["scalar-accuracy"]["y"]
train_acc = lambda d: train_acc_over_time(d)[-1]
Δ_acc_over_time = lambda d: (np.array(d["scalar-accuracy"]["y"]) - d["test_acc"])
Jaccard2last_max_over_time = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).max(0)
Jaccard2last_min_over_time = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).min(0)
Jaccard2last_mean_over_time = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]).mean(0)
Jaccard2last_mean_over_time_thres = lambda d: np.array(d["scalar2d-["+d["mode_data"]+"][sincelast][>T] JI(last,current)"]["z"]).mean(0)
Jaccard2last_median_over_time = lambda d: np.median(np.array(d["scalar2d-["+d["mode_data"]+"][sinceLast] JI(last,current)"]["z"]), 0)
deriv = lambda x: np.array(x)[1:]-np.array(x)[:-1]


cached_names = {}
def get_name(x):
    if x in cached_names:
        return cached_names[x]
    for k in globals():
        if globals()[k] == x:
            cached_names[x] = k
            return k


