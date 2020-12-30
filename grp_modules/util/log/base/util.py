import datetime
from os.path import basename
from html import escape
import git
from colored import fg, attr


def datestr_sort():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def datestr():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_git_revisions():

    # check repo for SHA and diffs
    repo = git.Repo(search_parent_directories=True)
    name = basename(repo.working_dir)
    sha = [repo.head.object.hexsha]
    diffs = [repo.git.diff('HEAD')]
    modules = [name]

    # check also submodules for SHAs and diffs
    if len(repo.submodules) > 0:
        modules += [s.name for s in repo.submodules]
        sha += [s.hexsha for s in repo.submodules]
        diffs += [s.module().git.diff('HEAD') for s in repo.submodules]

    return modules, sha, diffs


def training_header(state):
    gpu = ("gpu" + str(state.all["gpu"][0]) if len(state.all["gpu"]) == 1 else "multigpu(%s)" % ",".join(str(g) for g in state.all["gpu"])) if "gpu" in state.all else "cpu"
    s = [" ", "Experiment", state["tag"], "on", gpu, " "]
    seed_mode = "seed: %s " % state["seed"] if "seed" in state and state["seed"] >= 0 else "random mode"

    bar = "—" * len(" ".join(s))  # pylint: disable=blacklisted-name
    s[1] = s[1]
    s[2] = fg('red') + attr('bold') + s[2] + attr('reset')
    s[3] = attr('dim') + s[3] + attr('reset')
    s[4] = fg('red') + attr('bold') + s[4] + attr('reset')
    print()
    print(" ╭" + bar + "╮")
    print(" │" + " ".join(s) + "│", attr('dim') + seed_mode + attr('reset'))
    print(" ╰" + bar + "╯")
    if "record" in state and state["record"]:
        print(fg('red') + "     Recording Log-Calls" + attr('reset'))


def html_summary(state, event):
    html_repostate = "<ul style='list-style: circle'>" + ("".join("<li style='margin:0 3em;'>%s:%s:<code>%s</code></li>" % (name, "clean" if len(diff) == 0 else "<b>diverged</b>", sha[:7]) for (name, diff, sha) in state["repository_state"])) + "</ul>"
    html_loaded_modules = "<ul style='list-style: circle'>" + ("".join("<li style='margin:0 3em;'>%s</li>" % s for s in state["loaded_modules"])) + "</ul>"
    html_env = "<ul style='list-style: circle'>" + ("".join("<li style='margin:0 3em;'>%s: <code>%s</code></li>" % (name, ver) for (name, ver) in [("python", state["python"]), ("pytorch", state["pytorch"])])) + "</ul>"
    html_prepend = """
<h1>Experiment on %s</h1>
<h1 style="font-size:120%%; margin-top: -0.25em;">%s</h1>

<b>Repository Status:</b></br> %s </br></br>

<b>CLI-Call:</b></br> <code><pre>%s</pre></code> </br></br>

<b>Loaded Modules:</b></br> %s </br></br>

<b>Environment:</b></br> %s </br></br>
    """ % (state["date"], state["tag"], html_repostate, state["cli_overwrites"], html_loaded_modules, html_env)

    html_diffs = "\n".join("""
<h1>Repository Diffs</h1>
<b><b>%s</b>:</b></br> <code><pre>%s</pre></code> </br></br>
""" % (module, escape(diff)) for module, diff, sha in state["repository_state"])

    html_settings = html_prepend + "".join(event.settings_html())
    return html_settings, html_diffs


def plot_every(state, steps):
    return steps and state["main.current_batch"] % steps == 0
