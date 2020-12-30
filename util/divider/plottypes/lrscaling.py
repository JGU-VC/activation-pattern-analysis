# ============== #
# lr-scale plots #
# ============== #

def lr_scale_scalar_plots(expname, net):
    name_re = re.compile("%s-%s__%s\.json" % (word_re,word_re,value_re))
    def name_match_fn(d,m):
        d["expname"], d["net"], d["lrscale"] = m[1], m[2], m[3]
    data = get_data(name_re, name_match_fn)
    print(expname)
    dataset = datasets[expname.split("-")[-1]]
    print(dataset)

    L = locals()
    L.update(globals())
    # plt.style.use('ggplot')
    keys = list(data.keys())
    lrscales = sorted(list(set([d["lrscale"] for d in data.values()])), key=lambda s: 1/float(s[1:]) if s.startswith(":") else float(s))
    scatter = None
    # lrscales = lrscales[len(lrscales)//4:3*len(lrscales)//4:2]
    thresh0 = 20
    thresh = 30
    every = 4
    lrscales = list(filter(lambda s: int(s[1:])%every or int(s[1:])<=thresh0 if s.startswith(":") else int(s)%every or int(s)<=thresh0, lrscales))
    lrscales = list(filter(lambda s: int(s[1:])<=thresh if s.startswith(":") else int(s)<=thresh, lrscales))
    print(lrscales)

    C, ci = {}, 0
    for l in lrscales:
        # C[l] = np.log(1/float(l[1:]) if l.startswith(":") else float(l))
        C[l] = 1/float(l[1:]) if l.startswith(":") else float(l)
        ci += 1
    cmax = max(C.values())
    cmin = min(C.values())

    ax_x = 2
    # plots = [(train_H_mean,train_acc),(train_H_mean,test_acc),(train_step,train_H_mean_per_step),(train_step,Jaccard2last_max),(train_step,Jaccard2last_mean),(train_step,Jaccard2last_min)]
    # plots = [(train_H_correlation,train_acc),(train_step,Jaccard2last_mean)]
    # plots = [(train_H_mean,train_acc),(train_step,Jaccard2last_mean)]
    plots = [(train_H_mean,train_acc),(train_step,Jaccard2last_median)]
    # plots = [(train_H_mean,train_acc),(train_step,Jaccard2last_max)]
    # plots = [(train_H_mean,train_acc),(train_step,Jaccard2last_mean_thresh)]
    ax_y = len(plots)//ax_x
    fig, axs = plt.subplots(ax_y,ax_x, figsize=(10,2.5), gridspec_kw={'width_ratios': [1, 3]})

    data_keys = ["%s-%s__lrscale=%s.json" % (expname,net,l) for l in lrscales]
    data_keys = list(filter(lambda k: k in data, data_keys))



    for_csv = {
        "lr": [],
    }
    for plot1, plot2 in plots:
        for_csv[get_name(plot1)] = []
        for_csv[get_name(plot2)] = []

    for ax, (x_label,y_label) in zip(axs.T.flat,plots):
        ax.set_title(networks[net])
        if not x_label or not y_label:
            continue
        x = [x_label(data[key]) for key in data_keys]
        y = [y_label(data[key]) for key in data_keys]
        color = [C[data[key]["lrscale"]] for key in data_keys]
        label = [data[key]["lrscale"] for key in data_keys]

        # get correlation
        if x_label == train_H_correlation:
            x = np.stack(x).reshape([len(data_keys),-1])
            y = np.stack(y).reshape([-1,1])
            D = x
            # D = np.concatenate([x,y],1)
            D -= np.mean(D,0)
            pca = PCA(1)
            D = pca.fit_transform(D)
            x = D

        # c = plt.cm.viridis((color-cmin)/(cmax-cmin))
        scatter = ax.scatter(x,y,c=color,label=label)
        # ax.plot(x,y, '*:')#,c=c,label=l)

        camelnames = {
            "train_H_mean": "mean Activation Entropy",
            "train_acc": "Train Accuracy",
            "test": "Test Accuracy",
            "Jaccard2last_median": "median Jaccard Index",
            "train_step": "Training Time"
        }
        ax.set_xlabel(camelnames[get_name(x_label)])
        ax.set_ylabel(camelnames[get_name(y_label)])
        fig.subplots_adjust(left=0.1,right=1.1,bottom=0.1,wspace=0.3, hspace=None)

        hlines_at = set()

        for key, l, lr in zip(data_keys, label, color):
            c = plt.cm.viridis((lr-cmin)/(cmax-cmin))

            # any_len = len(any_d["scalar-loss"]["y"])
            # losses = np.stack([d["scalar-loss"]["y"] for d in data.values() if len(d["scalar-loss"]["y"]) == any_len])
            # jis = np.stack([Jaccard2last_mean_over_time(d) for d in data.values()])
            # ji_x = np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["x"],dtype=np.int)
            # ji_y = np.mean(np.array(any_d["scalar2d-[tm|trd][sinceLast] JI(last,current)"]["z"]),0)
            # loss_x = np.array(any_d["scalar-loss"]["x"],dtype=np.int)

            x2 = None
            y2 = None
            if x_label == train_H_mean:
                x2 = train_H_over_time(data[key])
            if x_label == train_step:
                x2 = train_step_over_time(data[key])
            if y_label == train_acc:
                y2 = train_acc_over_time(data[key])
            elif y_label == Δ_acc:
                y2 = Δ_acc_over_time(data[key])
            elif y_label == Jaccard2last_max:
                y2 = Jaccard2last_max_over_time(data[key])
                # y2 = deriv(y2)
                # x2 = x2[1:]
            elif y_label == Jaccard2last_min:
                y2 = Jaccard2last_min_over_time(data[key])
                # y2 = deriv(y2)
                # x2 = x2[1:]
            elif y_label == Jaccard2last_mean:
                y2 = Jaccard2last_mean_over_time(data[key])
            elif y_label == Jaccard2last_mean_thresh:
                y2 = Jaccard2last_mean_over_time_thres(data[key])
                # y2 = deriv(y2)
                # x2 = x2[1:]
            elif y_label == Jaccard2last_median:
                y2 = Jaccard2last_median_over_time(data[key])
            elif y_label == train_H_mean_per_step:
                # x2 = [0]+x2
                y2 = train_H_over_time(data[key])
            else:
                continue

            if x2 is not None and y2 is not None:
                f = len(y2)//len(x2)
                f2 = len(x2)//len(y2)
                if f >= 1:
                    # assert f*len(x2) == len(y2)
                    # y2 = y2[::f]
                    y2 = y2[::f][:len(x2)]
                if f2 >= 1:
                    # assert f2*len(y2) == len(x2)
                    # x2 = x2[::f2]
                    x2 = x2[::f2][:len(y2)]
                ax.plot(x2,y2,color=c,alpha=0.5)

                # save data to be stored in csv file
                if x_label == train_H_mean:
                    x2 = x2[1:]
                if y_label == train_acc:
                    y2 = y2[1:]
                if x_label == plots[0][0]:
                    for_csv["lr"].append(lr)
                for_csv[get_name(x_label)].append(x2)
                for_csv[get_name(y_label)].append(y2)


            if x_label == train_step:

                # determine learning rate drops
                scalarlr_tag = 'scalar-learning rate' if 'scalar-learning rate' in data[key] else 'scalar-learning rate (group 0)' 
                scalarlr_train_steps = np.array(data[key][scalarlr_tag]['x'])
                scalarlr_values = np.array(data[key][scalarlr_tag]['y'])
                scalarlr_changes_at = np.where(scalarlr_values[:-1] != scalarlr_values[1:])[0]

                scalarlr_changes_at = set(scalarlr_changes_at).difference(hlines_at)
                if len(scalarlr_changes_at):
                    print(scalarlr_changes_at)
                    ax.vlines(scalarlr_train_steps[list(scalarlr_changes_at)], ymin=0, ymax=1)
                hlines_at.update(scalarlr_changes_at)



    # f = fig.add_subplot(111, frameon=False)
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    # plt.grid(False)
    # plt.tight_layout()
    if scatter:
        plt.colorbar(scatter, ax=axs.ravel().tolist(), norm=matplotlib.colors.LogNorm())
    # plt.title("%s. (Colored by log(learning rate))" % networks[net])
    # plt.rcParams['figure.figsize'] = (10.0, 7.0)
    # plt.rcParams['axes.titlesize'] = "small"
    # font = {'family' : 'normal',
    #     'weight' : 'normal',
    #     'size'   : 12}
    # rc('font', **font)
    # plt.savefig("%s-%s.png" % (expname, net), dpi=200)
    plt.title(networks[net])
    for key, data in for_csv.items():
        if key == "lr":
            continue
        if key == "train_H_correlation":
            continue
        if key == "train_acc":
            continue
        np.savetxt("paper/lrscaling-%s/data/%s-%s.csv" % (dataset,net,key), np.stack(data).T, header=" ".join(str(s) for s in for_csv["lr"]), fmt=" ".join(['%.5f']*len(for_csv["lr"])))
    plt.savefig("paper/lrscaling-%s/img/lrscaling-%s.pdf" % (dataset,net))
    # plt.show()




