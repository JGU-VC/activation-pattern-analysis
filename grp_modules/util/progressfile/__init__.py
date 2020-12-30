from os import environ
from miniflask import outervar


def write_progress(state, tqdm_epoch=outervar):
    d = tqdm_epoch.format_dict
    elapsed = tqdm_epoch.format_interval(d['elapsed'])
    progress = "%.2f%%" % (d['n'] / d['total'] * 100)
    remaining = tqdm_epoch.format_interval(d["elapsed"] * (d["total"] or 0) / max(d["n"], 1))
    with open(state["progressfile"], "w") as f:
        f.write(progress + "\n")
        f.write(elapsed + "\n")
        f.write(remaining)


def register(mf):
    if "PROGRESSFILE" in environ:
        mf.register_event("before_epoch", write_progress, unique=False)
        mf.register_helpers({
            "progressfile": environ['PROGRESSFILE']
        })
