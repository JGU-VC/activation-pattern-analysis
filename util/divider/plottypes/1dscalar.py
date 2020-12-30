import re
import json
from subprocess import Popen, PIPE
import numpy as np

from util.names import Jaccard2last_mean_over_time, train_H_over_time
from util.extract import get_data, get_expname, compile_filename

def register(parser):
    parser.add_argument('files', type=str, nargs='+', help='number of files')
    parser.add_argument('scalarname', type=str, help='plotname')


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(plt, args):
    files = filter
    print(args.files)
    plotname = args.scalarname
    if plotname.endswith(".json") or plotname.endswith(".bin"):
        raise ValueError("No plotname specified.")

    for full_filename in args.files:
        expdir = "/".join(full_filename.split("/")[:-1])
        filename = full_filename.split("/")[-1]
        expname = filename[:-5]

        jq = lambda cmd: Popen("jq '%s' %s " % (cmd,full_filename), shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].decode('utf-8')
        jq_json = lambda cmd: json.loads(jq(cmd))
        jq_array = lambda cmd: np.array(jq_json(cmd))
        keys = jq_json('.jsons | keys')
        mode_data = re.compile(".*scalar2d-\[(\w+\|\w+)\].*").match(",".join(keys))[1]

        x_type = "%i"
        if "flr" in filename and "mcmc" in filename:
            name_re = compile_filename("flr-mcmcstats-{word}-{value}-{value}_{value}")
            def name_match_fn(d,m):
                d["net"], d["perlayer"], d["initlr"], d["seed"] = m[1], m[2], m[3], m[4]
            data = get_data(args.files, name_re, name_match_fn, expname=expdir+"/"+expname, exclude_unfinished=True, cache=True)
        else:
            # print(keys)
            test_acc = float(jq('.jsons["scalar-test_acc_1"].content.data[-1].y[-1]'))
            print(filename, test_acc)

        if plotname == "meanji":
            data = jq_json('.jsons["scalar2d-['+mode_data+'][sinceLast] JI(last,current)"].content.data[0]')
            x = data["x"]
            x = np.array(x, dtype=np.int)
            y = np.mean(data["z"],0)
        elif plotname == "dLdJI":
            any_d = next(iter(data.values()))
            any_len = len(any_d["scalar-loss"]["y"])
            losses = np.stack([d["scalar-loss"]["y"] for d in data.values() if len(d["scalar-loss"]["y"]) == any_len])
            jis = [Jaccard2last_mean_over_time(d) for d in data.values()]
            min_len = np.min([ji.shape[0] for ji in jis])
            jis = [ji[:min_len] for ji in jis]
            jis = np.stack(jis)
            ji_x = np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["x"],dtype=np.int)
            ji_y = np.mean(np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["z"]),0)
            ji_x = ji_x[:min_len]
            ji_y = ji_y[:min_len]
            # ji_y = np.stack([d["scalar-learning rate"]["y"] for d in data.values() if len(d["scalar-loss"]["y"]) == any_len])
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
            x_type = "%.8f"

            y = moving_average(y, n=25)
            x = moving_average(x, n=25)
        elif plotname == "dEdJI":
            any_d = next(iter(data.values()))
            any_len = len(any_d["scalar-loss"]["y"])
            losses = np.stack([d["scalar-loss"]["y"] for d in data.values() if len(d["scalar-loss"]["y"]) == any_len])
            jis = [train_H_over_time(d) for d in data.values()]
            min_len = np.min([ji.shape[0] for ji in jis])
            jis = [ji[:min_len] for ji in jis]
            jis = np.stack(jis)
            ji_x = np.array(any_d["scalar2d-[tm|trd] % max Entropy"]["x"],dtype=np.int)
            ji_y = np.mean(np.array(any_d["scalar2d-[tm|trd] % max Entropy"]["z"]),0)
            ji_x = ji_x[:min_len]
            ji_y = ji_y[:min_len]
            # ji_y = np.stack([d["scalar-learning rate"]["y"] for d in data.values() if len(d["scalar-loss"]["y"]) == any_len])
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
            x_type = "%.8f"

            y = moving_average(y, n=25)
            x = moving_average(x, n=25)
        else:
            data = jq_json(".jsons[\"scalar-"+plotname+"\"].content.data[0]")
            x = data["x"]
            y = data["y"]
        plt.plot(x,y)

        # np.savetext("paper/fantasticlr/data/%s.csv" % plotname)
        # np.savetxt("paper/fantasticlr-cifar10/data/%s-%s.csv" % (expname,plotname), np.array([x,y]).T, header="x y", fmt=" ".join([x_type,'%.8f']), comments="")

        y2 = moving_average(y, n=25)
        x2 = moving_average(x, n=25)
        # np.savetxt("paper/fantasticlr-cifar10/data/%s-%s-smooth.csv" % (expname,plotname), np.array([x2,y2]).T, header="x y", fmt=" ".join([x_type,'%.8f']), comments="")

        if plotname == "dLdJI" or plotname == "dEdJI":
            break

    fontsize = 2
    plt.tight_layout()
    # plt.legend()
    plt.title(plotname)
    # np.savetxt("/tmp/scalar1d-%s.txt" % (plotname), [x,y])
    # plt.savefig("paper/fantasticlr/img/scalar1d-%s.pdf" % (plotname))
    plt.show()
    # plt.savefig("paper/fantasticlr/img/scalar1d-%s.pdf" % (plotname))

    # save as csv
    # np.savetxt("paper/measures/data/%s-%s.csv" % (filename,plotname), data, header="x y z", fmt=" ".join(['%s','%s','%.8f']))



