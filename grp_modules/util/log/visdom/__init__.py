import sys
import socket
import tempfile
from subprocess import call, Popen
from time import sleep
from os import path, makedirs, remove
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

from colored import fg, attr
import visdom

# pylint: disable=E0401
from ..base.util import datestr_sort, training_header, html_summary
from .plot_img import plot_img
from .plot_bar import plot_bar
from .plot_scalar import plot_scalar
from .plot_scalars import plot_scalars
from .plot_scalar2d import plot_scalar2d
from .plot_pca import plot_pca
from .plot_static_hm import plot_static_hm
from .plot_scatter import plot_scatter
# pylint: enable=E0401


class NoneDict(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def text_summary(state, event):
    html_settings, html_diffs = html_summary(state, event)
    # save current settings in log
    state["vis"].text(text=html_settings, win="settings", env=state["tag"])
    state["vis"].text(text=html_diffs, win="diffs", env=state["tag"])


def ensure_visdom_server_running(logdir, port):

    # if not, first try to start with tmux
    print(fg('red') + "Starting visdom server in a tmux session..." + attr('reset'))
    ret = call(r"tmux new-session -A -s visdom 'visdom -env_path %s' \; detach" % logdir, shell=True)

    # if this command fails, start a visdom server on a new port
    # (in this way, no two calls can interfere with each other)
    if ret != 0:

        # get a free port & release it again
        print(fg('red') + "No tmux found. Looking for a new port to start visdom on.", attr('reset'))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        port = sock.getsockname()[1]
        sock.close()
        f, filename = tempfile.mkstemp()

        # start the visdom server directly
        print(fg('red') + ("Starting visdom server in a seperate python session on port %i ...\n\t- Exits when script finishes\n\t- Visdom server output in file: %s" % (port, filename)) + attr('reset'))
        _ = Popen(['visdom', '-env_path', logdir, '-port', str(port)], stdout=f, stderr=f)

    return port


def process(file, port):
    tag = path.basename(file)[:-4]

    # init visdom
    vis = visdom.Visdom(env=tag, port=port)

    # read all recorded commands and pass to corresponding real function
    print(fg('green') + "converting" + attr('reset'), tag)
    try:
        vis.replay_log(file)

        # save to json
        vis.save([tag])

        # remove when done
        remove(file)

    # pylint: disable=W0703
    except Exception:
        print(fg('red') + "converting of file %s failed" % file + attr('reset'))
        return False
    # pylint: disable=W0703
    print(fg('green') + "converting done" + attr('reset'), tag)
    return True


def init(state, event):

    if hasattr(event, 'main') and state["convert"]:
        raise ValueError("You are trying to convert a recording. However, you also sepcified a main loop. To convert, do not load any other modules besides this.")

    # feed visdom with all files to be converted
    if state["convert"]:

        def main():

            # silent visdom messages
            visdom.logger.setLevel(visdom.logging.CRITICAL)

            # get all recordings
            print(fg('red') + "Starting Conversion from logs.rec to actual logs." + attr('reset'))
            files = list(Path(".").rglob('*.rec'))
            print(len(files), "File(s) to be Converted")

            state["vis"] = visdom.Visdom(env=state["tag"], port=state["port"])
            if not preinitialized and not state["vis"].check_connection() and state["autostart"]:
                state["port"] = ensure_visdom_server_running(state["dir"], state["port"])

            with Pool(processes=cpu_count()) as pool:
                for _ in pool.imap_unordered(partial(process, port=state["port"]), files):
                    pass

            sys.exit(0)

        event.main = main

    # enable defaults, s.t. these settings appear in settings-lists
    state.default.update({state.module_id + "." + k: state[k] for k in ["python", "pytorch"]})

    # add list of loaded modules
    state["loaded_modules"] = list(event._mf.modules_loaded.keys())

    # check if user wishes to add random hash to log
    if state["tag"].endswith("!"):
        state["tag"] = state["tag"][:-1] + "-" + datestr_sort()

    # silent visdom messages
    visdom.logger.setLevel(visdom.logging.CRITICAL)

    # init visdom & check if server is available
    preinitialized = "vis" in state
    if not preinitialized:
        if state["record"]:
            # ensure directory exists
            if not path.exists(state["dir"]):
                makedirs(state["dir"])

            # precrete visdom (without server)
            state["vis"] = visdom.Visdom(env=state["tag"], log_to_filename=path.join(state["dir"], state["tag"] + ".rec"), offline=True)
            preinitialized = True
        else:
            state["vis"] = visdom.Visdom(env=state["tag"], port=state["port"])
    if not preinitialized and not state["vis"].check_connection() and state["autostart"]:
        state["port"] = ensure_visdom_server_running(state["dir"], state["port"])
        state["vis"] = visdom.Visdom(env=state["tag"], port=state["port"])
        sleep(2)

    text_summary(state, event)


def before_training(state):
    training_header(state)
    flush_log(state)


def flush_log(state, *args, **kwargs):
    """Flush visdom calls to `.json` file."""
    state["vis"].save([state["tag"]])
    return None, args, kwargs


def register(mf):
    mf.load("tensorhelpers")
    mf.load("..base")
    mf.set_scope("..")
    mf.register_defaults({
        "port": 8097,
        "steps.scalar": 100,
        "autostart": True,
        "convert": False,
        "record": False,
    })
    mf.register_helpers({
        "WINDOWS": NoneDict(),
        "Bar": {},
        "Scalar": {},
        "Scalar2D": {}
    })

    mf.register_event('init', init)
    mf.register_event('before_training', before_training)

    mf.register_event('after_epoch', flush_log)
    mf.register_event('after_training', flush_log)
    mf.register_event('after_testing', flush_log)
    mf.register_event('after_main', flush_log)

    mf.register_event('plot_img', plot_img)
    mf.register_event('plot_bar', plot_bar)
    mf.register_event('plot_scalar', plot_scalar)
    mf.register_event('plot_scalars', plot_scalars)
    mf.register_event('plot_scalar2d', plot_scalar2d)
    mf.register_event('plot_static_hm', plot_static_hm)
    mf.register_event('plot_pca', plot_pca)
    mf.register_event('plot_scatter', plot_scatter)
