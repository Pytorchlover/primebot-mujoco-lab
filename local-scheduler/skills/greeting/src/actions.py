from demo_backend import run_action


def say(context, text):
    return run_action(context, "greeting.say", {"text": text}, 0.1)
