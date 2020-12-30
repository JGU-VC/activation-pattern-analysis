import re
import json
from subprocess import Popen, PIPE
import numpy as np

from util.names import Jaccard2last_mean_over_time, train_H_over_time
from util.extract import get_data, get_expname, compile_filename

def register(parser):
    parser.add_argument('files', type=str, nargs='+', help='number of files')


# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n

def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot(plt, args):

    settings = set()

    name_re = compile_filename("flr-mcmcstats-{word}-{value}-{value}_{value}")
    name_re = compile_filename("flr-mcmcstats-{word}-{value}-{value}_{value}_{value}")
    def name_match_fn(d,m):
        d["net"], d["perlayer"], d["initlr"], d["seed"] = m[1], m[2], m[3], m[4]
        settings.add((d["net"], d["perlayer"], d["initlr"]))
    data = get_data(args.files, name_re, name_match_fn, exclude_unfinished=True, cache=True)

    settings = sorted(list(settings))

    # plotname == "dLdJI":
    for net, perlayer, initlr in settings:
        data_setting = dict(filter(lambda d: d[1]["net"] == net and  d[1]["perlayer"] == perlayer and d[1]["initlr"] == initlr, data.items()))

        any_d = next(iter(data_setting.values()))
        any_len = len(any_d["scalar-loss"]["y"])
        losses = np.stack([d["scalar-loss"]["y"] for d in data_setting.values() if len(d["scalar-loss"]["y"]) == any_len])
        jis = [Jaccard2last_mean_over_time(d) for d in data_setting.values()]
        min_len = np.min([ji.shape[0] for ji in jis])
        jis = [ji[:min_len] for ji in jis]
        jis = np.stack(jis)
        ji_x = np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["x"],dtype=np.int)
        ji_y = np.mean(np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["z"]),0)
        ji_x = ji_x[:min_len]
        ji_y = ji_y[:min_len]
        # ji_y = np.stack([d["scalar-learning rate"]["y"] for d in data_setting.values() if len(d["scalar-loss"]["y"]) == any_len])
        # jis = np.array([ji_y])
        # ji_x = any_d["scalar-learning rate"]["x"]
        loss_x = np.array(any_d["scalar-loss"]["x"],dtype=np.int)
        losses_var = losses.var(0)
        jis_mean = jis.mean(0)
        jis_mean = np.interp(loss_x, ji_x, jis_mean)
        x, y = loss_x[1:], losses_var[1:]/jis_mean[1:]**2
        # x = np.linspace(0.2*len(y),len(y),len(y))/len(y)
        # x = ji_y
        x = np.interp(loss_x, ji_x, ji_y)
        x = x[1:]

        y = np.mean(rolling_window(y, 20), axis=-1)
        x = np.mean(rolling_window(x, 20), axis=-1)

        plt.plot(x, y, label="%s %s %s" % (net, perlayer, initlr))

        np.savetxt("/tmp/flr-mcmcstats.csv", np.array([x,y]).T)
    # np.savetext("paper/fantasticlr/data/%s.csv" % plotname)
    # np.savetxt("paper/fantasticlr-cifar10/data/%s-%s.csv" % (expname,plotname), np.array([x,y]).T, header="x y", fmt=" ".join([x_type,'%.8f']), comments="")

    # y2 = moving_average(y, n=25)
    # x2 = moving_average(x, n=25)
    # np.savetxt("paper/fantasticlr-cifar10/data/%s-%s-smooth.csv" % (expname,plotname), np.array([x2,y2]).T, header="x y", fmt=" ".join([x_type,'%.8f']), comments="")

    fontsize = 2
    plt.tight_layout()
    plt.legend()
    # plt.title(plotname)
    # np.savetxt("/tmp/scalar1d-%s.txt" % (plotname), [x,y])
    # plt.savefig("paper/fantasticlr/img/scalar1d-%s.pdf" % (plotname))
    plt.show()
    # plt.savefig("paper/fantasticlr/img/scalar1d-%s.pdf" % (plotname))

    # save as csv
    # np.savetxt("paper/measures/data/%s-%s.csv" % (filename,plotname), data, header="x y z", fmt=" ".join(['%s','%s','%.8f']))



