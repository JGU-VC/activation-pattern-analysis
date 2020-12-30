import os
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn


def reset_checks(state):
    for mode_i, (mode, dataset) in enumerate(zip(state["net_mode"],state["dataset_mode"])):
        state["last_measured"][mode_i] = 0

def init(state,event):
    event.rng_save()

    if state["dummy"]:
        state["skip"] = True

    # generate new copy of net
    # nothing shall be done to the original net while evaluating
    # TODO: this is only needed for batch-statistics. can this be improved performance-wise?
    if "train" in state["net_mode"]:
        state["current_id"] = 0
        state["net_cached"] = event.send_net_to_device(event.init_net())

    # add queries
    queries = state["activation_stats.queries"]
    if "HA" in state["query"] or "MAL" in state["query"] or "HAgL" in state["query"]:
        query = {
            "save_current_counts": False,
            "with_grad": False,
            "with_labels": False,
            "histdecay": 1.0,
            "use_global_stats": False,
            "active": False,
            "additional_id": "validation",
        }
        queries.append(query)
    if "HAL" in state["query"] or "MAL" in state["query"] or "HAgL" in state["query"]:
        query = {
            "save_current_counts": False,
            "with_grad": False,
            "with_labels": True,
            "histdecay": 1.0,
            "use_global_stats": False,
            "active": False,
            "additional_id": "validation",
        }
        queries.append(query)
    state["queries"] = queries

    assert len(state["dataset_mode"]) == 1
    state["dataloader"] = event.dataloader(state["dataset_mode"][0], use_cache=False, deterministic=True, with_transform=False)

    # check if non-allowed mode used
    if len(set(state["net_mode"])) != len(state["net_mode"]):
        raise ValueError("Currrently only one dataset_mode per net_mode allowed.")

    # evaluate pretraned network for since_final mode
    if state["since_final"]:
        home = state["current_dir"] if "current_dir" in state else "."
        ckpt_file = os.path.join(home, "ckpts", state["tag"]+".ckpt")
        checkpoint = torch.load(ckpt_file)

        # sanity check
        # checkpoint = torch.load(ckpt_file)
        # state["main.net"].load_state_dict(checkpoint['state_dict'])

        # validate over full dataset
        for mode_i, (mode, dataset) in enumerate(zip(state["net_mode"],state["dataset_mode"])):

            if mode == "train":
                net = state["net_cached"]
                net.load_state_dict(checkpoint['state_dict'])
                net.train()

            relus = [m for m in net.modules() if "activation" in str(type(m))]
            for id, relu in enumerate(relus):
                for query in state["queries"]:
                    relu.reset_stats(query, only_global_stats=True)
                    relu.set_stats_active(query["id"], True)
            event.validate(valloader=state["dataloader"], evalmode=mode=="eval", plot=False, dummy=state["dummy"])
            for id, relu in enumerate(relus):
                for query in state["queries"]:
                    relu.set_stats_active(query["id"], False)
                q_id = state["queries"][0]["id"]
                relu.stats[q_id]["global_stats_final"] = relu.stats[q_id]["global_stats"]
                del relu.stats[q_id]["global_stats"]
            event.rng_restore()

    # event.set_seed()
    state["current_id"] = 0
    event.rng_restore()


def after_step(state, event, result=None, *args, **kwargs):
    event.rng_save()
    progress = state["main.current_batch"] / state["main.num_batches"]
    progress += state["main.current_epoch"]
    check = progress // state["every_epoch"]
    query_texts = {
        "HA": "Entropy",
        "HAL": "H(A,L)",
        "HAgL": "H(A|L)",
        "MAL": "MI(A;L)"
    }
    at_least_one_measured = False
    for mode_i, (mode, dataset) in enumerate(zip(state["net_mode"],state["dataset_mode"])):
        if check and check > state["last_measured"][mode_i] // state["every_epoch"] or state["last_measured"][mode_i] < 0:
            at_least_one_measured = True
            state["last_measured"][mode_i] = progress
            net = state["main.net"]
            title = "[%sm|%sd]" % (mode[0],dataset[:2])

            # get cached network if train mode is used
            # TODO: try this without net_cached (batch norms update gets in the way a.t.m.)
            if mode == "train":
                state["net_cached"].load_state_dict(state["main.net"].state_dict())
                net = state["net_cached"]
                net.train()

            # validate over full dataset
            relus = [m for m in net.modules() if "activation" in str(type(m))]
            for id, relu in enumerate(relus):
                for query in state["queries"]:
                    relu.reset_stats(query, only_global_stats=True)
                    relu.set_stats_active(query["id"], True)
            event.validate(valloader=state["dataloader"], evalmode=mode=="eval", plot=False, dummy=state["dummy"])
            for id, relu in enumerate(relus):
                for query in state["queries"]:
                    relu.set_stats_active(query["id"], False)
            if state["dummy"]:
                continue

            # plot
            n_cls = state["dataset.num_classes"]
            ε = 1e-14
            for id, relu in enumerate(relus):
                q_id = state["queries"][0]["id"]

                # get counts from relu
                counts         = relu.stats[q_id]["global_stats"]
                counts_labels  = relu.stats[state["queries"][1]["id"]]["global_stats"] if len(state["queries"]) > 1 else None
                hashmap_size   = len(counts)
                num_data       = counts.sum()
                counts_nonzero = counts[counts >= ε]
                max_counts     = counts.max()
                if counts_labels is not None:
                    counts_labels  = counts_labels[counts_labels >= ε]
                len_counts_nonzero = len(counts_nonzero)

                # ---------------- #
                # entropy measures #
                # ---------------- #

                for query in state["query"]:

                    # calculate entropy
                    max_H = event.calc_max_H_from_counts(counts_nonzero, num_data, relu.num_filters, counts_with_labels=counts_labels, mode=query, n_cls=n_cls)
                    entropy = event.calc_H_from_counts(counts_nonzero, num_data, counts_with_labels=counts_labels, mode=query, n_cls=n_cls)

                    # plot measures
                    query_text = query_texts[query]
                    event.optional.plot_scalar2d(entropy, id, title=title+" "+query_text)
                    event.optional.plot_scalar2d(entropy/max_H, id, title=title+" %% max %s" % query_text)
                    relu.entropy_perc = (entropy/max_H).item()
                    relu.entropy = entropy.item()
                    relu.max_H = max_H
                event.optional.plot_scalar2d(max_counts, id, title=title+" Count most frequent Pattern")
                event.optional.plot_scalar2d(1.0*max_counts/num_data, id, title=title+" % Count of most frequent Pattern")
                del counts_nonzero


                # ---------------- #
                # hashmap measures #
                # ---------------- #

                # histogram measures
                event.optional.plot_scalar2d(len_counts_nonzero/num_data.double(), id, title=title+" % Num Patterns")
                event.optional.plot_scalar2d(len_counts_nonzero/hashmap_size, id, title=title+" % Hashmap Filled")
                event.optional.plot_scalar2d(len_counts_nonzero, id, title=title+" Num Patterns")
                if not hasattr(relu, 'init_DS_act_stats'):
                    event.optional.plot_bar(num_data, id, title="Num Data")
                    event.optional.plot_bar(int(str(counts.dtype)[-2:]), id, title="Hashmap dtype")
                    event.optional.plot_bar(len(counts), id, title="Hashmap size")
                    relu.init_DS_act_stats = True

                if state["plot.since_final"] and "global_stats_final" not in relu.stats[q_id]:
                    raise ValueError("No final statistics found. Rerun with DS-act-stats.save_final module instead.")

                if state["plot.since_init"] and "global_stats_init" in relu.stats[q_id] or state["plot.since_last"] and "global_stats_last" in relu.stats[q_id]:
                    hasbin       = relu.stats[q_id]["global_stats"] > 0
                    hasbin_t     = relu.stats[q_id]["global_stats"] > state["threshold"]
                    hasbin_sum   = hasbin.sum()
                    hasbin_t_sum = hasbin_t.sum()

                # count changes since initialization
                if state["plot.since_init"] and "global_stats_init" in relu.stats[q_id]:
                    hasbin_init          = relu.stats[q_id]["global_stats_init"] > 0
                    assert len(hasbin) == len(hasbin_init)
                    same_as_init         = (hasbin_init & hasbin).sum()
                    hasbin_init_sum      = hasbin_init.sum()
                    changes_since_init   = ((~hasbin_init & hasbin).int() - (hasbin_init & ~hasbin).int()).sum()
                    del hasbin_init
                    hasbin_init_t        = relu.stats[q_id]["global_stats_init"] > state["threshold"]
                    same_as_init_t       = (hasbin_init_t & hasbin_t).sum()
                    hasbin_init_t_sum    = hasbin_init_t.sum()
                    changes_since_init_t = ((~hasbin_init_t & hasbin_t).int() - (hasbin_init_t & ~hasbin_t).int()).sum()

                    if state["plot.absolute_stats"]:
                        event.optional.plot_scalar2d(changes_since_init, id, title=title+"[sinceInit] Patterns Changed")
                    if state["plot.relative_stats"]:
                        event.optional.plot_scalar2d(1.0*same_as_init/(hasbin_sum + hasbin_init_sum - same_as_init), id, title=title+"[sinceInit] JI(init,current)")
                        event.optional.plot_scalar2d(1.0*(same_as_init_t+ε)/(hasbin_t_sum + hasbin_init_t_sum - same_as_init_t + ε), id, title=title+"[sinceInit][>T] JI(init,current)")
                        wJI = (torch.min(counts,relu.stats[q_id]["global_stats_init"]).sum().item()/torch.max(counts,relu.stats[q_id]["global_stats_init"]).sum().item())
                        event.optional.plot_scalar2d(1.0*wJI, id, title=title+"[sinceInit] wJI(init,current)")
                        event.optional.plot_scalar2d(1.0*wJI/state["every_epoch"], id, title=title+"[sinceInit] rwJI(init,current)")
                        event.optional.plot_scalar2d(1.0*changes_since_init/( len_counts_nonzero ), id, title=title+"[sinceInit] % Patterns Changed")

                # count changes since last statistics
                if state["plot.since_last"] and "global_stats_last" in relu.stats[q_id]:
                    hasbin_last          = relu.stats[q_id]["global_stats_last"] > 0
                    assert len(hasbin) == len(hasbin_last)
                    hasbin_last_sum      = hasbin_last.sum()
                    same_as_last         = (hasbin_last & hasbin).sum()
                    changes_since_last   = ((~hasbin_last & hasbin).int() - (hasbin_last & ~hasbin).int()).sum()
                    changeH_max = torch.log2(1.0*(hasbin | hasbin_last).sum())

                    # change entropy (not a real entropy)
                    # change = 1.0*(relu.stats[q_id]["global_stats_last"] - relu.stats[q_id]["global_stats"]).abs()
                    # hasbin_sum_weighted   = (change*hasbin).sum()
                    # hasbin_last_sum_weighted   = (change*hasbin_last).sum()
                    # same_as_last_weighted = (change*(hasbin_last & hasbin)).sum()

                    # kullback-leibler
                    # counts = relu.stats[q_id]["global_stats"]
                    # counts_last = relu.stats[q_id]["global_stats_last"]
                    # KL = (counts/num_data * torch.log2(counts/(counts_last+ε)+ε)).sum().item()
                    # KL2 = (counts_last/num_data * torch.log2(counts_last/(counts+ε)+ε)).sum().item()
                    # event.optional.plot_scalar2d(KL, id, title=title+"[sinceLast] KL1")
                    # event.optional.plot_scalar2d(KL2, id, title=title+"[sinceLast] KL2")

                    # weighted IJ
                    # JI_last_current_weighted = (1.0*same_as_last_weighted/(hasbin_sum_weighted + hasbin_last_sum_weighted - same_as_last_weighted)).item()
                    # relu.changeH = changeH.item()
                    # relu.changeH_perc = changeH.item()/changeH_max.item()
                    # relu.changeH_perc = JI_last_current_weighted

                    # weighted IJ
                    counts_curr = relu.stats[q_id]["global_stats"]
                    counts_last = relu.stats[q_id]["global_stats_last"]
                    JI_last_current_weighted = 1.0*torch.min(counts_curr,counts_last).sum().item()/torch.max(counts_curr,counts_last).sum().item()
                    relu.weighted_JI = JI_last_current_weighted
                    # relu.changeH = changeH.item()
                    # relu.changeH_perc = changeH.item()/changeH_max.item()
                    # relu.changeH_perc = JI_last_current_weighted

                    del hasbin_last
                    # change Entropy
                    # # changeH = 1.0*(relu.stats[q_id]["global_stats_last"] - relu.stats[q_id]["global_stats"]).abs()
                    # # changeH /= changeH.sum() + 1e-20
                    # # changeH = - (changeH * torch.log2(changeH + 1e-20)).sum()
                    # # changeH /= torch.max(relu.stats[q_id]["global_stats_last"], relu.stats[q_id]["global_stats"]) + 1e-20
                    # changeH = 1.0*(relu.stats[q_id]["global_stats_last"] - relu.stats[q_id]["global_stats"]).abs()
                    # changeH /= changeH.sum() + 1e-20
                    # # changeH = 1.0*(changeH - changeH.min()) / (changeH.max() - changeH.min())
                    # changeH = - (changeH * torch.log2(changeH + 1e-20)).sum()

                    hasbin_last_t        = relu.stats[q_id]["global_stats_last"] > state["threshold"]
                    same_as_last_t       = (hasbin_last_t & hasbin_t).sum()
                    hasbin_last_t_sum    = hasbin_last_t.sum()
                    changes_since_last_t = ((~hasbin_last_t & hasbin_t).int() - (hasbin_last_t & ~hasbin_t).int()).sum()

                    JI_last_current = (1.0*same_as_last/(hasbin_sum + hasbin_last_sum - same_as_last)).item()
                    JIT_last_current = (1.0*(same_as_last_t + ε)/(hasbin_t_sum + hasbin_last_t_sum - same_as_last_t + ε)).item()
                    relu.JI_last_current = JI_last_current
                    relu.JIT_last_current = JIT_last_current
                    if state["plot.absolute_stats"]:
                        event.optional.plot_scalar2d(changes_since_last, id, title=title+"[sinceLast] Patterns Changed")
                    if state["plot.relative_stats"]:
                        event.optional.plot_scalar2d(JI_last_current, id, title=title+"[sinceLast] JI(last,current)")
                        event.optional.plot_scalar2d(JIT_last_current, id, title=title+"[sincelast][>T] JI(last,current)")
                        event.optional.plot_scalar2d(JI_last_current_weighted, id, title=title+"[sinceLast] wJI(last,current)")
                        event.optional.plot_scalar2d(JI_last_current_weighted/state["every_epoch"], id, title=title+"[sinceLast] rwJI(last,current)")
                        event.optional.plot_scalar2d(1.0*changes_since_last/( len_counts_nonzero ), id, title=title+"[sinceLast] % Patterns Changed")
                    # event.optional.plot_scalar2d(changeH/changeH_max, id, title=title+"[sinceLast] % Weighted JI")
                        event.optional.plot_scalar2d(JI_last_current_weighted*relu.entropy, id, title=title+"[sinceLast] wJI·Entropy")
                        event.optional.plot_scalar2d(JI_last_current_weighted*relu.entropy_perc, id, title=title+"[sinceLast] wJI·relEntropy")

                if state["plot.since_final"] and "global_stats_final" in relu.stats[q_id] or state["plot.since_last"] and "global_stats_last" in relu.stats[q_id]:
                    hasbin       = relu.stats[q_id]["global_stats"] > 0
                    hasbin_t     = relu.stats[q_id]["global_stats"] > state["threshold"]
                    hasbin_sum   = hasbin.sum()
                    hasbin_t_sum = hasbin_t.sum()

                # count changes since final
                if state["plot.since_final"] and "global_stats_final" in relu.stats[q_id]:
                    hasbin_final          = relu.stats[q_id]["global_stats_final"] > 0
                    assert len(hasbin) == len(hasbin_final)
                    same_as_final         = (hasbin_final & hasbin).sum()
                    hasbin_final_sum      = hasbin_final.sum()
                    changes_since_final   = ((~hasbin_final & hasbin).int() - (hasbin_final & ~hasbin).int()).sum()
                    del hasbin_final
                    hasbin_final_t        = relu.stats[q_id]["global_stats_final"] > state["threshold"]
                    same_as_final_t       = (hasbin_final_t & hasbin_t).sum()
                    hasbin_final_t_sum    = hasbin_final_t.sum()
                    changes_since_final_t = ((~hasbin_final_t & hasbin_t).int() - (hasbin_final_t & ~hasbin_t).int()).sum()

                    if state["plot.absolute_stats"]:
                        event.optional.plot_scalar2d(changes_since_final, id, title=title+"[sinceFinal] Patterns Changed")
                    if state["plot.relative_stats"]:
                        event.optional.plot_scalar2d(1.0*same_as_final/(hasbin_sum + hasbin_final_sum - same_as_final), id, title=title+"[sinceFinal] JI(final,current)")
                        event.optional.plot_scalar2d(1.0*(same_as_final_t+ε)/(hasbin_t_sum + hasbin_final_t_sum - same_as_final_t + ε), id, title=title+"[sinceFinal][>T] JI(final,current)")
                        wJI = (torch.min(counts,relu.stats[q_id]["global_stats_final"]).sum().item()/torch.max(counts,relu.stats[q_id]["global_stats_final"]).sum().item())
                        event.optional.plot_scalar2d(1.0*wJI, id, title=title+"[sinceFinal] wJI(final,current)")
                        event.optional.plot_scalar2d(1.0*wJI/state["every_epoch"], id, title=title+"[sinceFinal] rwJI(final,current)")
                        event.optional.plot_scalar2d(1.0*changes_since_final/( len_counts_nonzero ), id, title=title+"[sinceFinal] % Patterns Changed")

                # save hashmap stats for change stats
                if state["plot.since_init"] and "global_stats_init" not in relu.stats[q_id]:
                    relu.stats[q_id]["global_stats_init"] = counts
                if state["plot.since_last"]:
                    if "global_stats_last" in relu.stats[q_id]:
                        del relu.stats[q_id]["global_stats_last"]
                    relu.stats[q_id]["global_stats_last"] = counts

    if at_least_one_measured:
        event.optional.actstats_done()
    event.rng_restore()
    return result, args, kwargs



def register(mf):
    mf.register_default_module("relu", required_event='activation_layer')
    mf.load("..activation_stats")
    mf.load("validate")
    mf.register_defaults({
        "threshold": 10,
        "query": ["HA"],
        # "query": ["HA","HAgL","HAL","MAL"],
        "net_mode": ["train"],
        "dataset_mode": ["train"],
        "every_epoch": 0.2,
        "plot.since_init": True,
        "plot.since_last": True,
        "plot.since_final": False,
        "plot.absolute_stats": True,
        "plot.relative_stats": True,
        "plot.test": True,
        "dummy": False,
    })
    mf.register_helpers({
        "last_measured": [-1,-1],
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

    mf.register_event('after_step', after_step, unique=False)
    mf.register_event('after_training', reset_checks, unique=False)
    mf.register_event('after_training', after_step, unique=False)
    mf.register_event('init', init, unique=False)
