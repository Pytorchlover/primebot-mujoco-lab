from demo_backend import run_action


def perform(context, seconds=2):
    return run_action(context, "dance.perform", {"seconds": seconds}, seconds)
