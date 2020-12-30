import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import inspect
from colored import fg, attr

from .util import DistributeGrad, bin2numbers, get_hist_torch

def init(state, event):

    def activation_wrap(*args, **kwargs):

        # wraps any activation function
        class ActivationStats(event.activation_layer_cls()):

            def __init__(self, inplace=False):

                # check if activation supports inplace=True
                kwargs = {}
                if 'inplace' in inspect.signature(super().__init__).parameters.keys():
                    kwargs['inplace'] = inplace
                elif inplace and not state["warning_applied"]:
                    print(fg('red')+attr('bold')+"Warning:"+attr('reset')+fg('red')+" PReLU cannot be used as an inplace activation."+attr('reset'))
                    state["warning_applied"] = True

                super().__init__(**kwargs)

                # create default stats
                self.stats = {}
                self.reset_stats()

                # register all relus in a list
                self.id = state["current_id"]
                state["current_id"] += 1
                state["relus"].append(self)

            # create stats from queries
            def set_stats_active(self, q_id, active=True):
                self.stats[q_id]["query"]["active"] = active

            # create stats from queries
            def reset_stats(self, q_id=None, only_global_stats=False):
                state["query_str"] = ""
                queries = state["queries"] if q_id is None else [state["queries"][q_id]] if isinstance(q_id, str) else [q_id]
                for q in queries:
                    q_id = event.activation_stats_query_to_str(q)
                    q["with_global_stats"] = q["histdecay"] <= 1 and q["histdecay"] > 0
                    q["active"] = q["active"] or True
                    q["id"] = q_id

                    if only_global_stats and q_id in self.stats:
                        self.stats[q_id]["query"] = q
                        self.stats[q_id]["global_stats"] = None
                    else:
                        self.stats[q_id] = {
                            "query": q,
                            "global_stats": None
                        }

                # check helper variables for cheapest computation for as much queries as possible
                self.any_with_grad = any(q["active"] and q["with_grad"] for q in state["queries"])
                self.any_with_labels = any(q["active"] and q["with_labels"] for q in state["queries"])
                self.any = any(q["active"] for q in state["queries"])
                self.needs_hist = any(q["active"] and not q["with_labels"] for q in state["queries"])
                self.needs_hist_with_labels = any(q["active"] and q["with_labels"] for q in state["queries"])


            # measure stats while forward pass
            def forward(self, input):
                if state["skip"]:
                    return super().forward(input)

                input_pass = input
                self.num_data = list(input_pass.shape)
                self.num_filters = self.num_data[1]
                self.num_data[1] = 1

                # get hashmap_size for current layer
                if hasattr(self, 'hashmap_size'):
                    hashmap_size = self.hashmap_size
                else:
                    if state["hashmap_size_mode"] == "constant":
                        hashmap_size = int(state["hashmap_size_factor"])
                    else:
                        hashmap_size = int(state["hashmap_size_factor_fn"](self.id, *input_pass.shape[1:]))
                    hashmap_size //= len(state["queries"])
                    self.hashmap_size = hashmap_size

                # get dtype for current layer
                if hasattr(self, 'hashmap_dtype'):
                    hashmap_dtype = self.hashmap_dtype
                else:
                    hashmap_dtype = state["hashmap_dtype_fn"](self.id, *input_pass.shape[1:])
                    self.hashmap_dtype = hashmap_dtype

                # flatten last dimension and convert to 0/1 tensor
                if self.any:
                    input_sign = F.relu(input.sign()).type(torch.bool)#.clone().detach().contiguous()

                    # map everything to R+, autograd will change directions accordingly
                    input = input.abs()

                # apply relaxation for gradiend descent
                # (only if gradient is needed)
                if self.any_with_grad:
                    input = event.actH_relaxation(input, altfn=lambda x: x)

                # get & reshape labels according to input.shape
                if self.any_with_labels:
                    labels = state["main.labels"]
                    for i in range(len(input_pass.shape)-1):
                        labels = labels.unsqueeze(i+1)
                    labels = labels.expand(self.num_data)
                else:
                    labels = None

                # reduce calls to get_hist_torch (slowest operation here)
                if self.needs_hist:
                    bins_to_act_indices_PRE, counts_PRE = get_hist_torch(input_sign, labels=None, N=hashmap_size, device=state["device"], dtype=hashmap_dtype)
                if self.needs_hist_with_labels:
                    bins_to_act_indices_PRE_LABELS, counts_PRE_LABELS = get_hist_torch(input_sign, labels=labels, N=hashmap_size, device=state["device"], dtype=hashmap_dtype)

                # now get the queried measures
                self.counts = {}
                for q_id, q_obj in self.stats.items():
                    q = q_obj["query"]
                    global_stats = q_obj["global_stats"]

                    # remove previous counts (free memory)
                    if q_id in self.counts:
                        del self.counts[q_id]

                    # skip inactive
                    if not q["active"]:
                        continue

                    # calculate histogram
                    # (use precomputed, use torch or with cpp_extension)
                    if q["with_labels"]:
                        bins_to_act_indices, counts = bins_to_act_indices_PRE_LABELS, counts_PRE_LABELS
                    else: # if not q["with_labels"]:
                        bins_to_act_indices, counts = bins_to_act_indices_PRE, counts_PRE

                    # clear unneeded memory
                    if not q["save_current_counts"]:
                        del bins_to_act_indices
                        if self.needs_hist:
                            del bins_to_act_indices_PRE
                        if self.needs_hist_with_labels:
                            del bins_to_act_indices_PRE_LABELS

                    # count = count*decay_rate + new_count
                    if q["histdecay"] > 0 and q["histdecay"] < 1 and global_stats is not None:
                        counts = global_stats*q["histdecay"] + counts

                    # merge with previous histogram
                    if q["histdecay"] > 0:

                        # register as global stat, if required and no stat register already
                        if q["with_global_stats"] and global_stats is None:
                            q_obj["global_stats"] = counts.clone()
                            global_stats = q_obj["global_stats"]
                            q_obj["relu_id"] = self.id

                        # merge
                        else:
                            global_stats += counts

                            # should global stats be used instead of batch-count?
                            if q["use_global_stats"]:
                                counts = global_stats

                    # apply gradient for counts if required
                    if q["save_current_counts"]:
                        no_zero = counts!=0
                        if q["with_grad"]:
                            counts = DistributeGrad.apply(input, counts, bins_to_act_indices)

                        # counts = counts.nonzero().squeeze()
                        # (does not work for floats)
                        counts = counts[no_zero]

                        # save result
                        self.counts[q_id] = counts

                # apply actual activation
                if state["bias"] != 0:
                    input_pass += state["bias"]
                return super().forward(input_pass)

        # event.activation_layer_cls = ActivationStats
        return ActivationStats(*args, **kwargs)

    event.activation_layer = activation_wrap

    # presets for hashmap_sizes
    dtype = state["hashmap_dtype"]
    if state["hashmap_size_mode"] == "auto":

        # resnet
        base_factor = state["hashmap_size_factor"]

        if "grp.model.resnet.blocks.basicblock_pyramid" in event._mf.modules_loaded and "grp.data.defaults.cifar10" in event._mf.modules_loaded:

            if state["num_blocks"] == [3, 3, 3]:
                #           0      1    2    3    4    5    6    7    8    9
                factors = [ 30, 200, 400, 150,   200, 200, 50,   50, 50, 50]
                net_factor = 1.6


        elif "grp.model.resnet.blocks.basicblock_fixup" in event._mf.modules_loaded and "grp.data.defaults.cifar10" in event._mf.modules_loaded:

            if state["num_blocks"] == [3, 3, 3]:
                #           0   1   2    3  4  5  6  7    8    9    10   11   12   13   14   15   16   17   18
                factors = [ 2, .25, 2, .25, 2, 2, 2, 150, 240, 25, 240,   5, 250, 150, 150, 150, 150, 150, 150]
                net_factor = 1.6

        elif "grp.model.resnet.cifar_variants.resnet_defaults" in event._mf.modules_loaded:

            # ======================= #
            # entropycurve-experiment #
            # ======================= #
            ls = 50 if "grp.data.defaults.cifar10" in event._mf.modules_loaded else 150

            # resnets
            if state["short"]:
                if state["num_blocks"] == [1, 1, 1]:
                    #           0    1  2  3     4  5    6
                    factors = [ 0.5, 1, 1, 90, 120, ls, ls ]
                    net_factor = 1
                elif state["num_blocks"] == [2, 2, 2]:
                    #           0     1    2  3  4  5      6   7   8    9   10  11  12  13
                    factors = [ 0.75, 1,   1, 1, 1, 100, 130, 150, 130, ls, ls, ls, ls, ls ]
                    net_factor = 1.9

                # resnets20
                elif state["num_blocks"] == [3, 3, 3]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12  13  14  15  16  17  18
                    factors = [ 0.75, 1,   1, 1, 1, 1, 1, 120, 150, 150, 150, 160, 150, ls, ls, ls, ls, ls, ls]
                    net_factor = 1.6

                elif state["num_blocks"] == [4, 4, 4]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12  13    14  15    16  17  18  19  20  21  22  23  24
                    factors = [ 0.5,  1,   1, 1, 1, 1, 1,  1,  1, 150,   150, 150, 150, 150, 150, 150, 150, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 2.1

                # resnets32
                elif state["num_blocks"] == [5, 5, 5]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12     13   14  15    16  17    18  19    20  21  22  23  24  25  26  27  28  29  30
                    factors = [ 0.75, 1,   1, 1, 1, 1, 1,  1,  1,   1,     1, 150,   150, 150, 150, 150, 150, 150, 150, 150, 150, ls, ls, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 2.3

                # resnets56
                elif state["num_blocks"] == [9, 9, 9]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12     13   14  15    16  17    18  19    20  21  22  23  24  25  26  27  28  29  30
                    factors = [ 1, *([1.5]*9*2), *([180]*9*2), *([int(1.5*ls)]*9*2), int(1.5*ls) ]
                    net_factor = 1.0
                    dtype = lambda id, c, w, h: 16 if id >= 20 else 32
                else:
                    raise ValueError("No hashmap_size_list given")

            # convnet20
            elif state["num_planes"] == [16, 32, 64]:
                if state["num_blocks"] == [1, 1, 1]:
                    #           0    1  2  3     4  5    6
                    factors = [ 0.75, 1, 1, 90, 120, ls, ls ]
                    net_factor = 1
                elif state["num_blocks"] == [2, 2, 2]:
                    #           0     1    2  3  4  5      6   7   8    9   10  11  12  13
                    factors = [ 0.75, 1,   1, 1, 1, 100, 130, 150, 130, ls, ls, ls, ls, ls ]
                    net_factor = 1.7
                elif state["num_blocks"] == [3, 3, 3]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12  13  14  15  16  17  18
                    factors = [ 0.75, 1,   1, 1, 1, 1, 1, 120, 150, 150, 150, 160, 150, ls, ls, ls, ls, ls, ls]
                    net_factor = 1.8
                elif state["num_blocks"] == [4, 4, 4]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12  13    14  15    16  17  18  19  20  21  22  23  24
                    factors = [ 0.5,  1,   1, 1, 1, 1, 1,  1,  1, 150,   150, 150, 150, 150, 150, 150, 150, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 2.0
                elif state["num_blocks"] == [5, 5, 5]:
                    #           0     1    2  3  4  5  6   7   8    9     10  11    12     13   14  15    16  17    18  19    20  21  22  23  24  25  26  27  28  29  30
                    factors = [ 0.5,  1,   1, 1, 1, 1, 1,  1,  1,   1,     1, 155,   155, 155, 155, 155, 155, 155, 155, 155, 155, ls, ls, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 2.2
                else:
                    raise ValueError("No hashmap_size_list given")

            # toynet20
            else:
                ls = 30 if "grp.data.defaults.cifar10" in event._mf.modules_loaded else 80
                if state["num_blocks"] == [1, 1, 1]:
                    #           0    1    2   3   4   5   6
                    factors = [ 20, 60, 100, 60, 120, ls, ls ]
                    net_factor = 1
                elif state["num_blocks"] == [2, 2, 2]:
                    ls = 20 if "grp.data.defaults.cifar10" in event._mf.modules_loaded else 80
                    #           0     1    2  3    4    5    6   7   8    9   10  11  12
                    factors = [ 15, 30, 100, 140, 170, 80,  80,  ls, ls,  ls, ls, ls, ls]
                    net_factor = 1.5
                elif state["num_blocks"] == [3, 3, 3]:
                    #           0    1    2  3    4      5   6   7    8    9  10  11  12  13  14  15  16  17  18
                    factors = [ 15, 60, 100, 140, 140, 180, 200, 80, 80,  80, 80, 80, 80, ls, ls, ls, ls, ls, ls]
                    net_factor = 1.7
                elif state["num_blocks"] == [4, 4, 4]:
                    #           0     1    2  3      4    5    6   7   8    9   10   11  12  13  14  15  16  17  18  19  20  21  22  23  24
                    factors = [ 15,  60, 100, 140, 140, 180, 200, 200, 240, 80, 80,  80, 80, 80, 80, 80, 80, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 1.7
                elif state["num_blocks"] == [5, 5, 5]:
                    #           0     1    2    3    4    5    6   7     8   9    10  11  12   13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30
                    factors = [ 15,  60, 100, 140, 140, 180, 200, 200, 240, 240, 240, 80, 80,  80, 80, 80, 80, 80, 80, 80, 80, ls, ls, ls, ls, ls, ls, ls, ls, ls, ls]
                    net_factor = 1.9
                else:
                    raise ValueError("No hashmap_size_list given")
        else:
            print("WARNING: No Hashmap-Size found. Consider defining the size.")
            state["hashmap_size_mode"] = "constant"

        if state["hashmap_size_mode"] == "auto":
            factors = np.array(factors).astype(np.float)
            factors /= factors.sum()
            factors *= net_factor

        state["hashmap_size_factor_fn"] = lambda id,d,w,h: factors[id] * base_factor
        if isinstance(dtype, int):
            state["hashmap_dtype_fn"] = lambda id,d,w,h: getattr(torch, "int"+str(dtype))
        elif isinstance(dtype, list):
            state["hashmap_dtype_fn"] = lambda id,d,w,h: getattr(torch, "int"+str(dtype[id]))
        else:
            state["hashmap_dtype_fn"] = lambda id,d,w,h: getattr(torch, "int"+str(dtype(id,d,w,h)))

    # convert dtype to dtype list
    # first value is used if smaller than second value many filters are used
    # if len(state["hashmap_dtype"]) != 3:
    #     state["hashmap_dtype"] = [state["hashmap_dtype"][0],0,state["hashmap_dtype"][0]]
    # state["hashmap_dtype"][0] = getattr(torch, state["hashmap_dtype"][0])
    # state["hashmap_dtype"][2] = getattr(torch, state["hashmap_dtype"][2])

def activation_stats_query_to_str(query):
    additional_id = query["additional_id"]+"-" if "additional_id" in query else ""
    return "%s%i-%i-%f-%i" % (additional_id, query["with_labels"], query["with_grad"], query["histdecay"], query["use_global_stats"])

def calc_max_H_from_counts(counts, total, num_filters, mode="HA", counts_with_labels=None, n_cls=None):
    if mode == "HA" or mode == "HAL" or mode == "MAL":
        max_H = max_HA = min(num_filters, torch.log2(total.double()))
    if mode == "HAgL" or mode == "MAL":
        max_H = max_HAgL = min(num_filters, torch.log2(total.double()/n_cls))
    if mode == "MAL":
        max_H = max_HA - max_HAgL
    return max_H

def calc_H_from_counts(counts, total, mode="HA", counts_with_labels=None, n_cls=None, residual_eps=0, estimated_p_cls=False):
    ε = 1e-14
    residual_H = torch.zeros((), dtype=counts.dtype, device=counts.device) if counts is not None else torch.zeros((), dtype=counts_with_labels.dtype, device=counts_with_labels.device)

    if mode == "HA" or mode == "HAL" or mode == "MAL":
        if mode == "HAL":
            counts = counts_with_labels

        # get entropy from these measures
        p_bin = counts.double() / (total+residual_eps)
        H = HA = -torch.sum(p_bin * torch.log2(p_bin))
        # alternative:
        # H = -torch.log2(torch.prod(torch.pow(p_bin,p_bin)))
        # H = -1/total*torch.sum(counts*torch.log2(counts)-counts*torch.log2(total))

        # add counter probability
        if residual_eps > 0:
            p_counter = 1-p_bin.sum()
            residual_H += -p_counter*torch.log2(p_counter+ε)

    if mode == "HLgA":
        raise ValueError("not implemented")

    if mode == "HAgL" or mode == "MAL":

        # get entropy from these measures
        p_bin = counts_with_labels.double() / (total+residual_eps)
        p_cls_true = 1/n_cls
        if estimated_p_cls:
            raise ValueError("To be checked: not implemented in current version")
            bin_cls = bins[:,-1]
            _, counts_cls = labels.unique(return_counts=True,sorted=True) 
            p_cls = counts_cls/total
            p_cls = p_cls[bin_cls]
        else:
            p_cls = p_cls_true
        H = HAgL = -torch.sum(p_bin * torch.log2(p_bin/p_cls))

        # residual_eps
        if residual_eps > 0:
            p_counter = 1-p_bin.sum()
            residual_H += -p_counter*torch.log2(p_counter/p_cls_true+ε)

    if mode == "MAL":
        H = HA - HAgL

    # make the value optimize residual_H, bug lets H ignore the additional value
    if residual_eps > 0:
        residual_H = residual_H * residual_eps
        H += residual_H - residual_H.detach()

    return H


    # ----- #
    # plots # (pca, entropy & current bias)
    # ----- #
    # if S("visdom.summary.numbins") and update_vis("entropyloop"):
    #     plot_vot(counts.max(), self.id, GLOBAL["total_i"], "max_count")
    #     if num_features > 20:
    #         min_val = counts.sum()
    #     else:
    #         min_val = min(counts.sum(), 2 ** num_features)
    #     plot_vot(len(counts) / min_val, self.id, GLOBAL["total_i"], "numbins_perc")
    #     plot_vot(len(counts), self.id, GLOBAL["total_i"], "numbins")
    #     plot_vot(len(self.stats), self.id, GLOBAL["total_i"], "numbins_total")
    #
    # if S("visdom.summary.dimused") and update_vis("entropyloop"):
    #     net = flatten(input).contiguous()
    #     net_sign = torch.sign(net).int()
    #     act_sum = net_sign.sum(0, keepdim=True).abs()
    #     dim_used = (act_sum != num_data) & (act_sum == act_sum) & (
    #                 act_sum != 0)  # possible values: NaN, [0, max]
    #     plot_vot((dim_used).float().sum().item() / dim_used.shape[1], self.id, GLOBAL["total_i"],
    #              "dim_used_perc")

# # plot
# if state["plot.every"] == 0 or state.all["step"] % state["plot.every"] == 0:
#     if state["plot.stats"] and self.stats is not None:
#         event.optional.plot_scalar2d(len(self.stats), self.id, title="Number of Patterns")
#         event.optional.plot_scalar2d(entropy_fromhash2count(self.stats)/self.max_H_full, self.id, title="% of full-stats Entropy")


def after_init_net(event, state, net, *args, **kwargs):

    # resnet20 has a uneven number of activations
    if "grp.model.resnet.default" in event._mf.modules_loaded:
        state["current_id"] += 1

    return net, args, kwargs

def register(mf):
    mf.register_default_module("relu", required_event='activation_layer')
    mf.register_defaults({
        "device": "same",
        "hashmap_size_factor": 5e8,
        "hashmap_size_mode": "auto",
        "hashmap_dtype": 32,
        "bias": 0.0,
        # "hashmap_size_mode": "constant",
    })
    mf.register_helpers({
        "skip": False, # for turning off counting completely
        "queries": [
        # example query
        # {
        #     "with_grad": True,
        #     "with_labels": True,
        #     "histdecay": 1.0,
        #     "use_global_stats": True, # use collected stats if exist
        # }
        ],
    })

    mf.register_helpers({
        "relus": nn.ModuleList(),
        "current_id": 0,
        "warning_applied": False
    }, parsefn=False)
    mf.register_event('init', init)
    mf.register_event('DistributeGrad', DistributeGrad, unique=True)
    mf.register_event('activation_stats_query_to_str', activation_stats_query_to_str, unique=True)
    mf.register_event('calc_H_from_counts', calc_H_from_counts, unique=True)
    mf.register_event('calc_max_H_from_counts', calc_max_H_from_counts, unique=True)
    mf.register_event('after_init_net', after_init_net, unique=False)
