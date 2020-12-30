import sys

import torch

from .util import plot_every, datestr, get_git_revisions  # pylint: disable=E0401


def register(mf):
    mf.set_scope("..")  # one up
    # log also the projects & subprojects state
    modules, shas, diffs = get_git_revisions()
    mf.register_defaults({
        "steps": 0,
        "dir": "./logs",
        "tag": "default"
    })
    mf.register_helpers({
        "repository_state": list(zip(modules, diffs, shas)),
        "git_diffs": diffs,
        "cli_overwrites": " ".join(sys.argv),
        "date": datestr(),
        "python": sys.version.replace("\n", " "),
        "pytorch": torch.__version__,
        "loaded_modules": ""
    })
    mf.register_event('plot_every', plot_every, unique=True)
