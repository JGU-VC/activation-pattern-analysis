import torchvision.transforms as transforms


def dataset_transform(state, event, splitname):
    del event  # unused
    T = []

    if splitname == "train":
        if state["flip"]:
            T.append(
                transforms.RandomHorizontalFlip(),
            )
        if state["flipvertical"]:
            T.append(
                transforms.RandomVerticalFlip(),
            )
        if state["cropsize"] > 0:
            T.append(
                transforms.RandomCrop(state["cropsize"], padding=state["croppadding"]),
            )
        if state["rotationdeg"] > 0:
            T.append(
                transforms.RandomRotation(state["rotationdeg"]),
            )
        if state["colorjitter"]:
            T.append(
                transforms.ColorJitter(),
            )

    return T


def register(mf):
    mf.register_defaults({
        "cropsize": 32,
        "croppadding": 4,
        "flip": True,
        "flipvertical": False,
        "colorjitter": False,
        "rotationdeg": 0
    })
    mf.register_event('dataset_transform', dataset_transform)
