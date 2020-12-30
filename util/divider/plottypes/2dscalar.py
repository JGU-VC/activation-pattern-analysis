import sys
import re
import numpy as np
from os import makedirs
from os.path import sep as path_sep
from colored import fg, attr

from util.names import networks, datasets, train_H_plot, test_acc
from util.extract import get_data, get_expname, compile_filename

def register(parser):
    parser.add_argument('--label', type=str, help='label')
    parser.add_argument('file', type=str, nargs=1, help='number of files')
    parser.add_argument('scalarname', type=str, help='plotname')


def plot(plt, args):
    plotname = args.scalarname
    expdir = "/".join(args.file[0].split("/")[:-1])
    filename = args.file[0].split("/")[-1]
    expname = filename.split("_")[0]
    expname2 = "_".join(filename.split("/")[-1].split("_")[1:])

    name_re = compile_filename("{word}-{word}_{word}(_.*|\.json)")
    def name_match_fn(d,m):
        d["expname"], d["mode"], d["net"] = m[1], m[2], m[3]
    data = get_data(args.file, name_re, name_match_fn, exclude_unfinished=False, cache=True)

    # print(data.keys())
    try:
        d = data[filename]
    except:
        print(fg('red'),filename,"not found",attr('reset'))
        return

    real_plotname = "scalar2d-[%s]%s" % (d["mode_data"], plotname if plotname.startswith("[") else " "+plotname)
    # print(d.keys())
    # print(d[plotname].keys())

    x, y, z, z_min, z_max = [d[real_plotname][k] for k in ["x","y","z","zmin","zmax"]]
    x, y, z = np.array(x), np.array(y), np.array(z)
    z_min, z_max = z.min(), z.max()

    print(fg('green'),plotname, z_min, z_max,attr('reset'))
    print(filename, "Accuracy",test_acc(d))
    if plotname == "Entropy":
        cmap = 'magma'
        cmap = 'inferno'
        z_min, z_max = 0, 21.30
    elif plotname == "% max Entropy":
        cmap = 'magma'
        cmap = 'inferno'
        # z_min, z_max = 0, 21.30
    elif plotname == "[sinceLast] % Patterns Changed":
        z_mean = z.mean(1, keepdims=True)
        z_max = z.max(1, keepdims=True)
        z_min = z.min(1, keepdims=True)
        # z = z / z_mean
        # z = (z - z_min)/ (z_max - z_min)

        # z_helper = z[:,0:]
        # z_min, z_max = z_helper.min(), z_helper.max()

        # z_min, z_max = -10, 4
        z_min, z_max = -0.5, 0.5
        # z = np.sign(z)*np.log(np.abs(z)+1e-7)
        # z_min, z_max = z.min(), z.max()
        # z_min, z_max = -1, 1
        # cmap = 'rainbow'
        print(z_min, z_max)
        cmap = 'viridis'
    elif plotname == "[sinceLast] JI(last,current)":
        cmap = 'viridis'
        z_min, z_max = 0, 1
    elif plotname == "[sinceInit] JI(init,current)":
        cmap = 'viridis'
        # z = np.log(z+1e-4)
        z_min, z_max = z.min(), z.max()
        # print(z_min, z_max)
        # z_min, z_max = -8, -0.05
    elif plotname == "% Num Patterns":
        cmap = 'twilight_shifted'
        z_min, z_max = 0, 1
    elif plotname == "% Count of most frequent Pattern":
        cmap = 'RdBu_r'
        z_min, z_max = 0, 0.376202
    elif plotname == "% Hashmap Filled":
        cmap = 'RdBu_r'
        # z_min, z_max = 0, 0.376202
    else:
        cmap = 'RdBu'
    plotname = plotname.replace("%","percent")
    filename = filename.replace(".json","")

    # plt.plot(x,y,z)
    # plt.imshow(a, cmap='hot', interpolation='nearest')
    fontsize = 2
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    p = ax.pcolormesh(x, y, z, shading='nearest', cmap=cmap, vmin=z_min, vmax=z_max, linewidth=0, rasterized=True)
    # ax.set_xlabel('x-label', fontsize=fontsize)
    # ax.set_ylabel('y-label', fontsize=fontsize)
    # ax.set_title('Title', fontsize=fontsize)
    # ax.set_xticklabels(x, fontsize=fontsize)
    # ax.set_yticklabels(y, fontsize=fontsize)
    ax.set_xticklabels([], fontsize=fontsize)
    ax.set_yticklabels([], fontsize=fontsize)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.colorbar(p)

    if not args.save:
        plt.show()
        return

    if not args.label:
        raise ValueError("Label missing")
    label = args.label
    netname = networks[d["net"]]

    makedirs("paper_plots/measures-%s/img/" % label, exist_ok=True)
    # plt.savefig("paper_plots/measures%s/img/%s-%s.pdf" % (label,filename,plotname))

    # save as csv
    x, y = np.meshgrid(np.array(x).astype(np.int),np.array(y).astype(np.int))
    z = np.array(z)
    data = np.stack([x,y,z]).reshape([3,-1]).T
    makedirs("paper_plots/measures-%s/data/%s/%s" % (label, d["mode"], netname), exist_ok=True)
    print("paper_plots/measures-%s/data/%s/%s/%s-%s.csv" % (label,d["mode"],netname,filename,plotname))
    np.savetxt("paper_plots/measures-%s/data/%s/%s/%s-%s.csv" % (label,d["mode"],netname,filename,plotname), data, header="x y z", fmt=" ".join(['%s','%s','%.8f']))



