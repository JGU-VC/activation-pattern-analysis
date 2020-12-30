
def register(mf):
    mf.register_event('scheduler_step', lambda: True, unique=False)
