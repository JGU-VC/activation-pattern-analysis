import sys
from colored import fg, attr


def before_backward(state, event, total_loss, *args, **kwargs):

    # save initial loss & exit if everything fails
    if state["initial_loss"] is None:
        state["initial_loss"] = total_loss.item()
    if total_loss > state["initial_loss"] * state["divergence_threshold"]:
        print("\n" * 5 + fg('red') + "Stopped due to divergent loss.\n\tLoss started with: %f\n\tLoss threshold to exit: %f\n\tLast loss before exis: %f" % (state["initial_loss"], state["initial_loss"] * state["divergence_threshold"], total_loss) + attr('reset'))
        if state["finalize-on-exit"]:
            event.optional.finalize()
        sys.exit(state["exit_code"])

    return [total_loss, *args], kwargs


def register(mf):
    mf.register_defaults({
        "divergence_threshold": 1.5,
        "finalize-on-exit": False,
        "exit_code": 1
    })
    mf.register_helpers({
        "initial_loss": None,
    })

    mf.register_event('before_backward', before_backward, unique=False)
