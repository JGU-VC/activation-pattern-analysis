
# ============== #
# acc vs entropy #
# ============== #
# name: "${expname}_${net}_${norm}_${activation}_${optimizer}_lr=${learning_rate}_wd=${weight_decay}_mom=${momentum}_${scheduler}_bias=${bias}"
def acc_vs_entropy():
    name_re = re.compile(("_".join(["{word}"]*5+["{value}"]*3+["{word}","{value}"])+".*").format(word=word_re,value=value_re))
    def name_match_fn(d,m):
        d["expname"], d["net"], d["norm"], d["activation"], d["optimizer"], d["learning_rate"], d["weight_decay"], d["momentum"], d["scheduler"], d["bias"] = m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10]
    data = get_data(name_re, name_match_fn)

    ax_x = 2
    ax_y = 1
    fig, axs = plt.subplots(ax_y,ax_x, figsize=(10,5))
    axs = axs.flat
    global_color = "net"
    global_color2 = "net"
    plots = [(train_H_mean,train_acc, global_color),(train_H_mean,test_acc, global_color)]#, (train_H_mean,train_acc, global_color2),(train_H_mean,test_acc, global_color2)]
    for ax, (x_label,y_label, c_label) in zip(axs,plots):
        x = [x_label(d) for d in data.values()]
        y = [y_label(d) for d in data.values()]
        labels = [d[c_label] for d in data.values()]
        labels_unique = list(set(labels))
        colors = [labels_unique.index(c) for c in labels]
        scatter = ax.scatter(x, y, c=colors, label=labels)
        ax.set_xlabel(get_name(x_label))
        ax.set_ylabel(get_name(y_label))
        ax.set_title('colored by %s' % (c_label))
        ax.legend(scatter.legend_elements()[0], labels_unique, loc="lower right")
    plt.show()


