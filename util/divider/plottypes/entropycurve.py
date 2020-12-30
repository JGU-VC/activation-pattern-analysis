import sys
import re
import numpy as np
from os import makedirs

from util.names import datasets, test_acc, learning_rate_per_step
from util.extract import get_data, get_expname, compile_filename

from util.names import train_H_plot
from util.names import train_maxH_plot

def register(parser):
    parser.add_argument('--label', type=str, help='label')
    parser.add_argument('files', type=str, nargs='+', help='number of files')
    parser.add_argument('--measure-points', type=int, nargs='+', help='number of files')
    parser.add_argument('--max', action='store_true', default=False, help='Save maximal entropy or Save entropy')

def plot(plt, args):
    # name_re = compile_filename("_".join(["{word}"]*5+["{value}"]*3+["{word}","{value}"]+["{value}","{word}"]))
    name_re = compile_filename("-".join(["{word}"]*2)+"_"+"_".join(["{value}"]*2))
    def name_match_fn(d,m):
        d["expname"], d["net"], d["depth"], d["seed"] = m[1], m[2], m[3], m[4]
        # d["expname"], d["net"], d["norm"], d["activation"], d["optimizer"], d["learning_rate"], d["weight_decay"], d["momentum"], d["scheduler"], d["bias"], d["depth"], d["seed"] = m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], int(m[12][4:])
    data = get_data(args.files, name_re, name_match_fn, exclude_unfinished=True)

    label = args.label
    if label and "." in label or (not label and args.save):
        if args.save:
            raise ValueError("label not specified")
            sys.exit(1)
    print("label", label)

    get_H = train_maxH_plot if args.max else train_H_plot

    # for debugging text
    any_d = data[list(data.keys())[0]]
    global_x = np.array(get_H(any_d)["x"]).astype(np.int)
    global_lrs = np.array(learning_rate_per_step(any_d)["y"]).astype(np.double)
    global_lrs_time = np.array(learning_rate_per_step(any_d)["x"]).astype(np.int)

    # get queried measure points
    # (if no measure points given, measure at 0, 1, -1 and also before biggest jump)
    times = args.measure_points
    if len(times) == 0:
        # times = [0, 1, -1]
        times = [0, 5, 13, 14, 21]

    global_lrs_time = [global_lrs[global_lrs_time <= t][-1] for t in global_x[times]]
    print("Queried steps:", global_x[times])
    print("With learning rates:", )
    print("Possible steps:", global_x)

    ax_x = 3
    ax_y = len(times)
    fig, axs = plt.subplots(ax_y,ax_x, figsize=(14,8))
    global_color = "net"

    keys_normalized = sorted(list(set(["_".join(k.split("_")[:-2]) for k in data.keys()])))
    keys_normalized[0], keys_normalized[1] = keys_normalized[1], keys_normalized[0]

    # convnet
    plot_later = []
    max_depth = 1
    max_val = 0
    for key, axs_row in zip(keys_normalized,axs.T):
        data_net = dict(filter(lambda d : d[0].startswith(key), data.items()))
        depths = sorted(list(set([int(d["depth"]) for d in data_net.values()])), reverse=True)
        for c, depth in enumerate(depths):
            data_depth = dict(filter(lambda d : d[1]["depth"] == str(depth), data_net.items()))
            any_d = data_depth[list(data_depth.keys())[0]]
            net_name = any_d["net"]
            global_y = get_H(any_d)["y"]
            real_depth = len(global_y)
            max_depth = max(max_depth, real_depth)

            # get combined dataset shape (seeds, depth, time)
            try:
                results = np.stack([get_H(d)["z"] for d in data_depth.values()])
                all_x = np.stack([get_H(d)["x"] for d in data_depth.values()]).astype(np.int)
                all_y = np.stack([get_H(d)["y"] for d in data_depth.values()]).astype(np.int)
            except:
                print("Some shapes do not match")
                print("Please check:")
                for key, d in data_depth.items():
                    print(key, "z-shape:", np.array(get_H(d)["z"]).shape, "x-shape:", np.array(get_H(d)["x"]).shape, "y-shape:", np.array(get_H(d)["y"]).shape)
                sys.exit(1)
            acc = np.mean([test_acc(d) for d in data_depth.values()])
            print(net_name, depth, acc)

            # assert x and y axes match
            assert (all_x == all_x[0]).all()
            assert (all_y == all_y[0]).all()
            max_val = max(max_val, results.max())

            for time, ax in zip(times, axs_row):

                result_at = results[:,:,time]

                # max/min
                rmin = result_at.min(0)
                rmax = result_at.max(0)
                rmean = result_at.mean(0)
                rstd = result_at.std(0)
                top = rmean + rstd
                bottom = rmean - rstd
                # top = rmax
                # bottom = rmin

                ax.fill_between(np.arange(1,result_at.shape[1]+1), top, bottom, alpha=0.4)
                # ax.set_title('%s after %s seen examples' % (net_name, global_x[time]), x=0.5, y=0)
                ax.netname = net_name
                # ax.depth = depth
                ax.time = global_x[time]
                # ax.acc = acc

                data_csv = np.stack([np.arange(real_depth).astype(np.int), rmean, rstd, rmin, rmax]).T
                # makedirs("paper/entropycurve-%s/data/" % label, exist_ok=True)
                # np.savetxt("paper/entropycurve-%s/data/%s-depth=%i-at=%s.csv" % (label,net_name, real_depth, global_x[time]), data_csv, header="layer mean std min max", fmt=" ".join(['%i']+['%.5f']*4))

                plot_later.append((ax, rmean))


    for ax, mean in plot_later:
        ax.set_ylim(0,1.05*max_val)
        ax.grid(True, axis='y', linestyle='--', color='black', alpha=0.2)
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=False) # labels along the bottom edge are off
        ax.plot(np.arange(1,len(mean)+1),mean,".-")
    for ax in axs[-1].flat:
        ax.set_xticks(np.arange(1,max_depth+1))
        ax.set_xlabel("Layer")
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True) # labels along the bottom edge are off
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_minor_locator(plt.MaxNLocator(33))
    for ax in axs.T[0].flat:

        print(ax.time)
        if ax.time == 0:
            ax.set_ylabel("Activation Entropy [%%]\nat Initialization" % (100.0*int(ax.time)/global_x[-1]))
        elif ax.time <= 10000:
            ax.set_ylabel("Activation Entropy [%%]\nat %i Training steps" % (int(ax.time)//128))
        elif ax.time == global_x[-1]:
            ax.set_ylabel("Activation Entropy [%%]\nafter Training" % (100.0*int(ax.time)/global_x[-1]))
        else:
            ax.set_ylabel("Activation Entropy [%%]\nat %.1f%% into Training" % (100.0*int(ax.time)/global_x[-1]))
        ax.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=True,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelleft=True) # labels along the bottom edge are off

    plt.tight_layout()
    if args.save:
        makedirs("paper/entropycurve-%s/img/" % label, exist_ok=True)
        print("saving to 'paper_plots/entropycurve-%s/img/entropycurve%s.pdf'" % (label,"-max" if args.max else ""))
        plt.savefig("paper_plots/entropycurve-%s/img/entropycurve%s.pdf" % (label,"-max" if args.max else ""))
    else:
        plt.show()

