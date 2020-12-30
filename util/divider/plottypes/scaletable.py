
# ====================== #
# lr/bs/mom-scale table  #
# ====================== #

def scale_table():
    name_re = re.compile(("_".join(["{word}"]*5+["{value}"]*3+["{word}","{value}"]+["{value}"]*3)+".*\.json").format(word=word_re,value=value_re))
    def name_match_fn(d,m):
        d["expname"], d["net"], d["norm"], d["activation"], d["optimizer"], d["learning_rate"], d["weight_decay"], d["momentum"], d["scheduler"], d["bias"], d["lrscale"], d["momscale"], d["bsscale"] = m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13]
    data = get_data(name_re, name_match_fn)

    scalar2d = "scalar2d-[tm|trd][sinceLast] JI(last,current)"
    # scalar2d = "scalar2d-[tm|trd] % max Entropy"
    # scalar2d = "scalar2d-[tm|trd][sinceLast][>T] JI(last,current)"
    data = dict(filter(lambda elem: scalar2d in elem[1], data.items()))
    keys = list(data.keys())

    lrscales = sorted(list(set([d["lrscale"] for d in data.values()])), key=lambda s: float(s))
    bsscales = sorted(list(set([d["bsscale"] for d in data.values()])), key=lambda s: float(s))
    momscales = sorted(list(set([d["momscale"] for d in data.values()])), key=lambda s: float(s))

    # ax_x = 7
    # ax_y = 7
    # plots = np.random.choice(keys,ax_x*ax_y)
    # fig, axs = plt.subplots(ax_y,ax_x, figsize=(10,5))

    vars = ["lrscales", "momscales"]
    # vars = ["lrscales", "bsscales"]
    # vars = ["momscales", "bsscales"]
    var_x = lrscales if "lrscales" in vars else momscales
    var_y = bsscales if "bsscales" in vars else momscales
    str_x = "learning rate scaling" if "lrscales" in vars else "momentum scaling"
    str_y = "batch size scaling" if "bsscales" in vars else "momentum scaling"
    lr_str = "%s" if "lrscales" in vars else "1.0"
    mom_str = "%s" if "momscales" in vars else "1.0"
    bs_str = "%s" if "bsscales" in vars else "1.0"
    ax_x = len(var_x)
    ax_y = len(var_y)

    plots = ["scalings-c10_convnet_batchnorm_relu_sgd_lr=std_wd=std_mom=std_multistep_bias=std_lrscale=%s_momscale=%s_bsscale=%s.json" % (lr_str,mom_str,bs_str) % (l,m) for m in var_y for l in var_x]
    # plots = ["scalings-c10_resnet20,basicblock_pre_batchnorm_relu_sgd_lr=std_wd=std_mom=std_multistep_bias=std_lrscale=%s_momscale=%s_bsscale=%s.json" % (lr_str,mom_str,bs_str) % (l,m) for l in var_x for m in var_y]
    fig, axs = plt.subplots(ax_y,ax_x, figsize=(10,5))


    min_z = min([np.array(data[key][scalar2d]["z"]).min() for key in plots if key in data])
    max_z = max([np.array(data[key][scalar2d]["z"]).max() for key in plots if key in data])
    for ax, key in zip(axs.flat,plots):
        if key not in data:
            print(key,"not found")
            continue
        d = data[key]
        ax.pcolormesh(d[scalar2d]["x"],d[scalar2d]["y"],d[scalar2d]["z"], vmin=min_z, vmax=max_z)
        # ax.set_title("lr=%s mom=%s bs=%s" % (d["lrscale"], d["momscale"], d["bsscale"]))
        ax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,         # ticks along the top edge are off
            right=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False
        ) # labels along the bottom edge are off


    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    for i in range(ax_x):
        plt.setp(axs[-1, i], xlabel=var_x[i])
    for i in range(ax_y):
        plt.setp(axs[i, 0], ylabel=var_y[i])
    plt.grid(False)
    plt.xlabel(str_x)
    plt.ylabel(str_y)
    plt.title("Value: \"%s\" for %s-vs-%s" % (scalar2d, str_x, str_y))
    plt.show()



