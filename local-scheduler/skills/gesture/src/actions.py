from demo_backend import run_action


def perform(context, name):
    return run_action(context, "gesture." + name, {"name": name}, 0.25)
