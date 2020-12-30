import os
import re
import sys
import json
from tqdm import tqdm
import pickle
import numpy as np
from pathlib import Path
from os import path
from subprocess import Popen, PIPE


# helpers
word_re = "([A-Za-z0-9\-,.:]+)"
value_re = "[A-Za-z0-9\-]+="+word_re


def compile_filename(re_str):
    return re.compile((re_str+".json").format(word=word_re,value=value_re))


def get_expname(fileglob):
    expname = fileglob.replace("*","").split("/")[-1]
    if expname.endswith("_"):
        expname = expname[:-1]
    return expname


def get_data(files, name_re, name_match_fn, exclude_unfinished=True, cache=True):
    meta = {"files": []}
    data = {}
    plots = []

    # filter .bin files
    files = list(filter(lambda f: not f.endswith(".bin"), files))

    common_prefix = os.path.commonprefix(files)
    dest = path.dirname(common_prefix)
    bin_file = "%s.bin" % common_prefix
    if cache and path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            meta, data = pickle.load(f)

    # query asks for subset
    if len(meta["files"]) > 0 and set(files) < set(meta["files"]):
        print("queried subset")
        diff = set(data.keys()) - set([f.split(os.path.sep)[-1] for f in files])
        for file in diff:
            if file in meta["files"]:
                meta["files"].remove(file)
            if file in data:
                del data[file]

    # if not same list, recreate database
    elif meta["files"] != files:
        meta = {}
        meta["files"] = files
        for file in tqdm(files, desc="Reading Data from Files", dynamic_ncols=True):
            jq = lambda cmd: Popen("jq '%s' %s " % (cmd,file), shell=True, stdout=PIPE, stderr=PIPE).communicate()[0].decode('utf-8')
            jq_json = lambda cmd: json.loads(jq(cmd))
            jq_array = lambda cmd: np.array(jq_json(cmd))
            try:
                keys = jq_json('.jsons | keys')
                keys.remove('diffs')
                mode_data = re.compile(".*scalar2d-\[(\w+\|\w+)\].*").match(",".join(keys))[1]
            except:
                print("skipping file %s. (seems incomplete)" % file)
                continue

            if not plots:
                plots = keys

            # show possible plots if no argument given
            # if plot.startswith("key"):
            #     print(jq(".jsons | keys"))
            #     sys.exit(0)

            # get data
            try:
                train_acc = float(jq('.jsons["scalar-accuracy"].content.data[-1].y[-1]'))
                test_acc = float(jq('.jsons["scalar-test_acc_1"].content.data[-1].y[-1]'))
            except:
                train_acc = 0
                test_acc = 0
                if exclude_unfinished:
                    continue

            train_H = jq_array('.jsons["scalar2d-['+mode_data+'] % max Entropy"].content.data[0].z | transpose | .[-1]')
            test_H = jq_array('.jsons["bar-(Test) % max Entropy"].content.data[0].y')
            # test_H = 0

            d = {
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_H": train_H,
                "test_H": test_H,
                "mode_data": mode_data
            }

            for p in plots:
                if p == "settings":
                    continue
                d[p] = jq_json(".jsons[\"%s\"].content.data[0]" % p)

            data[path.basename(str(file))] = d

        if os.path.isfile(bin_file):
            os.remove(bin_file)
        if cache:
            with open(bin_file, 'wb') as f:
                pickle.dump((meta,data),f,protocol=pickle.HIGHEST_PROTOCOL)

    # add measures
    for name, d in data.items():
        name = name.replace("basicblock_","basicblock")
        m = name_re.match(name)
        name_match_fn(d,m)

    for d in data.values():
        if "mode_data" not in d:
            d["mode_data"] = "tm|trd"

    return data


