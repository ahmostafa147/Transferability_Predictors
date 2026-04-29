import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def train_from_scratch(train_loader, test_loader, num_classes, epochs=30, lr=0.01, save_path=None):
    torch.manual_seed(42)
    model = models.resnet18(weights=None, num_classes=num_classes).to(DEVICE, memory_format=torch.channels_last)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    use_amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for ep in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}"):
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
    model.eval()
    correct = total = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(DEVICE, non_blocking=True).long()
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return correct / total
