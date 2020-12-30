

def dataset_transform(state, event, splitname):
    del state, event, splitname  # unused
    return []


def register(mf):
    mf.register_event('dataset_transform', dataset_transform)
