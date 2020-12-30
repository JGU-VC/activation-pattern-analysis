import re
from colored import attr, fg


def init(state):
    varid_list = state.all.keys()
    for key, val in zip(state["scale"][::2], state["scale"][1::2]):

        r = re.compile(r"^(.*\.)?%s$" % key)
        found_varids = list(filter(r.match, varid_list))
        for varid in found_varids:
            if isinstance(state[varid], float):
                if val.startswith("/") or val.startswith(":"):
                    _val = 1 / float(val[1:])
                else:
                    _val = float(val)
                if _val != 1.0:
                    print(fg('yellow') + "scaling:", varid, "=", state[varid], "->", state[varid] * _val, attr('reset'))
                    state[varid] = state[varid] * _val
            elif isinstance(state[varid], int):
                if val.startswith("/") or val.startswith(":"):
                    _val = 1 / float(val[1:])
                else:
                    _val = float(val)
                if _val != 1.0:
                    print(fg('yellow') + "scaling:", varid, "=", state[varid], "->", int(state[varid] * _val), attr('reset'))
                    state[varid] = int(state[varid] * _val)
            else:
                raise ValueError("Variable '%s' has type '%s', which is not supported for value scaling." % (varid, str(type(state[varid]))))


def register(mf):
    mf.register_event('init', init)
    mf.register_defaults({
        "scale": [str]
    })
