import torchvision
import torchvision.transforms as transforms


def get_data(state, event, splitname, with_transform=True):

    # get optional transforms
    if with_transform:
        transform = [t for tr in event.optional.dataset_transform(splitname) for t in tr]
    else:
        transform = []

    # data specific transform
    normalize = transforms.Normalize(mean=state["mean"], std=state["std"])
    transform += [transforms.ToTensor(), normalize]

    # compose dataset
    transform = transforms.Compose(transform)
    if splitname == "train":
        return torchvision.datasets.CIFAR10(root=state["dataset_root"], train=True, download=state["download"], transform=transform)
    if splitname == "test":
        return torchvision.datasets.CIFAR10(root=state["dataset_root"], train=False, download=state["download"], transform=transform)
    if splitname == "val":
        return None

    raise ValueError("splitname '%s' not known." % splitname)


def register(mf):
    mf.set_scope("..")
    mf.register_helpers({
        "dataset_root": "../datasets/cifar10",
        "num_classes": 10,
        "num_channels": 3,
        "subsets": ["train", "test"],

        # src: https://github.com/facebookarchive/fb.resnet.torch/issues/180
        "mean": [0.49139969, 0.48215842, 0.44653092],
        "std": [0.24703318, 0.24348606, 0.26158884]
    })
    mf.register_defaults({
        "download": True
    })
    mf.register_event('dataset', get_data, unique=True)
