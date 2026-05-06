import os
import numpy as np
import torch
from tqdm import tqdm
from torchvision import models
from data import get_loader

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
FEAT_DIR = "results/features"
os.makedirs(FEAT_DIR, exist_ok=True)


def build_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval().to(DEVICE, memory_format=torch.channels_last)
    for p in model.parameters():
        p.requires_grad = False
    feats = []

    def hook(_, __, out):
        feats.append(out.squeeze(-1).squeeze(-1).float().cpu().numpy())

    h = model.avgpool.register_forward_hook(hook)
    return model, feats, h


def extract_features(loader):
    model, feats, h = build_model()
    labels = []
    use_amp = DEVICE == "cuda"
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in tqdm(loader, desc="extract"):
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            model(x)
            labels.append(np.asarray(y))
    h.remove()
    F = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    return F, y


def imagenet_probs(F):
    fc = models.resnet18(weights="IMAGENET1K_V1").fc
    with torch.no_grad():
        return torch.softmax(fc(torch.from_numpy(F).float()), dim=1).numpy()


def cached_features(name, split, batch_size=512):
    path = f"{FEAT_DIR}/{name}_{split}.npz"
    if os.path.exists(path):
        d = np.load(path)
        return d["F"], d["y"]
    loader = get_loader(name, split, batch_size=batch_size, shuffle=False)
    F, y = extract_features(loader)
    np.savez_compressed(path, F=F, y=y)
    return F, y
