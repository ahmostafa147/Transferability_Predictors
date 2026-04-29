import torch
from torch.utils.data import DataLoader, Subset, IterableDataset
from torchvision import datasets, transforms
import numpy as np

SEED = 42
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
ROOT = "./data"

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

eval_gray_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_gray_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def _label_int(y):
    return int(y[0])


def _pick(augment, gray):
    if gray:
        return train_gray_tf if augment else eval_gray_tf
    return train_tf if augment else eval_tf


def _medmnist(name, split, augment):
    import medmnist
    from medmnist import INFO
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    tf = _pick(augment, info["n_channels"] == 1)
    ds = DataClass(split=split, transform=tf, target_transform=_label_int,
                   download=True, size=224)
    ds.targets = np.array(ds.labels).flatten()
    return ds


def get_dataset(name, split, augment=False):
    train = split == "train"
    tf = _pick(augment, False)
    gtf = _pick(augment, True)
    if name == "cifar10":
        return datasets.CIFAR10(ROOT, train=train, download=True, transform=tf)
    if name == "stl10":
        return datasets.STL10(ROOT, split="train" if train else "test", download=True, transform=tf)
    if name == "svhn":
        return datasets.SVHN(ROOT, split="train" if train else "test", download=True, transform=tf)
    if name == "eurosat":
        ds = datasets.EuroSAT(ROOT, download=True, transform=tf)
        n = len(ds)
        g = torch.Generator().manual_seed(SEED)
        idx = torch.randperm(n, generator=g).tolist()
        cut = int(0.8 * n)
        return Subset(ds, idx[:cut] if train else idx[cut:])
    if name == "pathmnist":
        return _medmnist("pathmnist", "train" if train else "test", augment)
    if name == "dermamnist":
        return _medmnist("dermamnist", "train" if train else "test", augment)
    if name == "imagenet_ref":
        return _hf_imagenet_val(10000, tf)
    raise ValueError(name)


class _HFImageNetVal(IterableDataset):
    def __init__(self, n, transform):
        self.n = n
        self.transform = transform

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset("imagenet-1k", split="validation", streaming=True).take(self.n)
        for ex in ds:
            img = ex["image"].convert("RGB")
            yield self.transform(img), int(ex["label"])


def _hf_imagenet_val(n, tf):
    return _HFImageNetVal(n, tf)


def get_loader(name, split, batch_size=128, shuffle=None, augment=False, num_workers=8):
    ds = get_dataset(name, split, augment=augment)
    if isinstance(ds, IterableDataset):
        return DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=True)
    if shuffle is None:
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True,
                      persistent_workers=num_workers > 0, prefetch_factor=4)


def num_classes(name):
    return {"cifar10": 10, "stl10": 10, "svhn": 10, "eurosat": 10,
            "pathmnist": 9, "dermamnist": 7}[name]
