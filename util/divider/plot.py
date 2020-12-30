#!/bin/python
import sys
import re
import json
from os import makedirs
from os.path import sep as path_sep
from colored import fg, attr
import matplotlib
from subprocess import Popen, PIPE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.decomposition import PCA
import matplotlib.transforms as mtransforms
from argparse import ArgumentParser
from pkgutil import iter_modules
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir)
import plottypes


# filter
# data = dict(filter(lambda elem: "sgd" in str(elem[0]), data.items()))
# data = dict(filter(lambda elem: elem[1]["train_acc"] > 0.2, data.items()))
# data = dict(filter(lambda elem: elem[1]["net"] == "convnet", data.items()))


if __name__ == "__main__":

    types = ["'%s'" % m.name for m in iter_modules(plottypes.__path__)]
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument('plotname', type=str, help='one of '+", ".join(types))

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    plotname = sys.argv[1]

    try:
        plot = __import__("plottypes.%s" % plotname, fromlist=["plottypes"])
    except ImportError:
        # Display error message
        print("Plotname '%s' not known." % plotname)
        sys.exit(0)

    # register plot argument parser
    plot.register(parser)
    parser.add_argument('--save', action='store_true', default=False, help='Saves picture. (By default it shows it on the screen).')
    args = parser.parse_args()

    # plot it!
    plot.plot(plt, args)

    # from old code
    # expname = get_expname(sys.argv[1])
    # if expname.startswith("get2d"):
    #     filename = sys.argv[2]
    #     plotname = sys.argv[3]
    #     get2d(filename, plotname)
    # elif expname.startswith("get1d"):
    #     filenames = sys.argv[2:-1]
    #     plotname = sys.argv[-1]
    #     get1d(filenames, plotname)
    # elif expname.startswith("as200"):
    #     acc_vs_entropy()
    # elif expname == "scalings-c10_":
    #     scale_table()
    # elif expname.startswith("lrscaling-"):
    #     if len(sys.argv) <= 1:
    #         raise ValueError("specify also the net to filter from")
    #     lr_scale_scalar_plots(expname, sys.argv[2])
    # elif expname.startswith("entropycurve-"):
    #     entropycurve()
    # else:
    #     print("Experiment not known registered with any plot.")
