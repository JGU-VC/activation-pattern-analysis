import re
from colored import attr, fg


def init(state, event):
    state["change_done"] = [False] * (len(state["change"]) // 3)

    # divide into change-requests for per_step-event and for per_epoch-event.
    for after_time, var, val in zip(state["change"][::3], state["change"][1::3], state["change"][2::3]):
        try:
            after_time = int(after_time)
            state["change_epoch"] += [after_time, var, val, False]
        except ValueError:
            after_time = float(after_time)
            if after_time.is_integer():
                state["change_epoch"] += [after_time, var, val, False]
            else:
                state["change_steps"] += [after_time, var, val, False]

    # miniflask-magic: remove the event, just in case event.step/after_time has already been called
    # reregister the events afterwards
    if state["change_steps"]:
        event.step  # pylint: disable=W0104
        del event.step
        del event.after_step
        state["mf"].register_event('after_step', after_step, unique=False)
    if state["change_epoch"]:
        event.after_epoch  # pylint: disable=W0104
        del event.after_epoch
        state["mf"].register_event('after_epoch', after_epoch, unique=False)


def after_epoch(state, *args, **kwargs):
    del args, kwargs
    do_change(state, state["change_epoch"])


def after_step(state, res, *args, **kwargs):
    do_change(state, state["change_steps"])
    return res, args, kwargs


def do_change(state, change):
    # get current progress
    progress = state["main.current_batch"] / state["main.num_batches"]
    progress += state["main.current_epoch"]

    varid_list = state.all.keys()
    for i, (after_time, var, val, done) in enumerate(zip(change[::4], change[1::4], change[2::4], change[3::4])):
        if progress >= after_time and not done:

            # mark as done
            change[i * 4 + 3] = True  # pylint: disable=W0104

            if var in state["keys"]:
                found_varids = state["keys"][var]
            else:
                r = re.compile(r"^(.*\.)?%s$" % var)
                found_varids = list(filter(r.match, varid_list))
                state["keys"][var] = found_varids
            for varid in found_varids:
                if isinstance(state[varid], float):
                    val = float(val)
                elif isinstance(state[varid], bool):
                    val = val.lower() == "true" or val[0].lower() == "t"
                elif isinstance(state[varid], int):
                    val = int(val)
                else:
                    raise ValueError("Variable '%s' has type '%s', which is not supported for change-at module." % (varid, str(type(state[varid]))))
                print(fg('yellow') + "changing:", varid, "=", state[varid], "->", val, "(at progress %f)" % progress, attr('reset'))
                state[varid] = val


def register(mf):
    mf.register_event('init', init, unique=False)
    mf.register_defaults({
        "change": [str]
    })
    mf.register_helpers({
        "change_steps": [],
        "change_epoch": [],
        "keys": {},
        "mf": mf
    })
