import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.resnet_cifar import ResNet18_CIFAR

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_loaders(data_dir, batch_size, num_workers=2):
    train_tfms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tfms)
    test_set  = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tfms)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def adjust_learning_rate(optimizer, epoch, base_lr):
    # Multi-step schedule as a simple default
    milestones = [60, 120, 160]
    gamma = 0.2
    lr = base_lr
    for m in milestones:
        if epoch >= m:
            lr *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += acc1 * bs
        n += bs
        pbar.set_postfix(loss=running_loss/n, acc=running_acc/n)
    return running_loss / n, running_acc / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_acc += acc1 * bs
        n += bs
    return running_loss / n, running_acc / n

def save_checkpoint(state, ckpt_dir, name):
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(state, os.path.join(ckpt_dir, name))

def parse_args():
    parser = argparse.ArgumentParser(description="ResNet-18 on CIFAR-10 (from scratch)")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 download/cache directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision training")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18_CIFAR(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size, num_workers=args.num_workers)

    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best_acc={best_acc:.2f})")

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, args.lr)
        print(f"Epoch {epoch+1}/{args.epochs}  lr={lr:.5f}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.2f} | Test: loss={val_loss:.4f} acc={val_acc:.2f}")

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "args": vars(args),
        }, args.checkpoint_dir, "last.pth")
        if is_best:
            save_checkpoint({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }, args.checkpoint_dir, "best.pth")

    print(f"Training complete. Best Top-1 Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
