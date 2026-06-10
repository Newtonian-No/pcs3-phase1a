#!/usr/bin/env python3
"""
PC-S³ Step 1: Training recipe upgrade — same Phase 1b architecture,
strong augmentations + 300 epoch AdamW to check baseline ceiling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse, json, time, copy, math
from pathlib import Path
from pcn import make_model as make_model_v1

# ═══════════════════════════════════════════════════════════════
# Data with strong augmentation
# ═══════════════════════════════════════════════════════════════

def get_cifar100_aug(data_dir='./data', batch_size=128, num_workers=4):
    """CIFAR-100 with RandAugment + Mixup/CutMix support."""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.25),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# Mixup / CutMix
# ═══════════════════════════════════════════════════════════════

def rand_bbox(size, lam):
    W, H = size  # explicit ints from .size()[-2:]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def mixup_cutmix(x, y, alpha_mix=0.8, alpha_cutmix=1.0, prob=1.0):
    if alpha_mix <= 0 and alpha_cutmix <= 0:
        return x, y, y, 1.0
    if torch.rand(1) > prob:
        return x, y, y, 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    if torch.rand(1) < 0.5 and alpha_mix > 0:
        lam = torch.distributions.Beta(alpha_mix, alpha_mix).sample().item()
        lam = max(lam, 1 - lam)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    elif alpha_cutmix > 0:
        lam = torch.distributions.Beta(alpha_cutmix, alpha_cutmix).sample().item()
        lam = max(lam, 1 - lam)
        bbx1, bby1, bbx2, bby2 = rand_bbox((x.size(-2), x.size(-1)), lam)
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        mixed_x = x.clone()
        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    return x, y, y, 1.0

def soft_cross_entropy(logits, y_a, y_b, lam, label_smoothing=0.1):
    """Mixup/CutMix loss with label smoothing."""
    n_classes = logits.size(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    with torch.no_grad():
        ya_smooth = (1 - label_smoothing) * nn.functional.one_hot(y_a, n_classes).float() + label_smoothing / n_classes
        yb_smooth = (1 - label_smoothing) * nn.functional.one_hot(y_b, n_classes).float() + label_smoothing / n_classes
        target = lam * ya_smooth + (1 - lam) * yb_smooth
    return -(target * log_probs).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    @torch.no_grad()
    def apply(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, backup):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(backup[n])


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            x, ya, yb, lam = mixup_cutmix(x, y, args.mixup_alpha, args.cutmix_alpha)
        else:
            ya, yb, lam = y, y, 1.0

        optimizer.zero_grad()
        logits = model(x)

        if lam < 1.0:
            loss = soft_cross_entropy(logits, ya, yb, lam, args.label_smoothing)
        else:
            if args.label_smoothing > 0:
                loss = soft_cross_entropy(logits, ya, yb, lam, args.label_smoothing)
            else:
                loss = nn.functional.cross_entropy(logits, y)

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_one_variant(mode, train_loader, test_loader, device, args):
    model = make_model_v1(mode, num_classes=100, d_model=args.d_model)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    print(f"\n{'='*60}")
    print(f"  PC-S³ Step 1: mode = {mode.upper()}")
    print(f"  Params: {params:,}  |  epochs={args.epochs}  |  batch={args.batch_size}")
    print(f"  Optim: AdamW(lr={args.lr}, wd={args.weight_decay})")
    print(f"  Aug: RA(N=2,M=9) + Mixup(α={args.mixup_alpha}) + CutMix(α={args.cutmix_alpha}) + RE")
    print(f"{'='*60}")

    best_acc = 0
    best_epoch = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        if ema:
            ema.update()
            ema.apply()
        test_loss, test_acc = evaluate(model, test_loader, device)
        if ema:
            ema.restore({n: p.data.clone() for n, p in model.named_parameters()})

        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, "test_acc": test_acc})

        lr_now = optimizer.param_groups[0]['lr']
        marker = " ★" if test_acc == best_acc else ""
        print(f"  Ep {epoch:3d} | lr {lr_now:.1e} | train loss {train_loss:.4f} | test loss {test_loss:.4f} acc {test_acc:.4f}{marker}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min. Best: {best_acc:.4f} @ epoch {best_epoch}")

    result = {
        "mode": mode,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "n_params": params,
        "time": elapsed,
        "history": history,
    }
    return result


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all', help='vanilla|concat|delta|B|C|concat_shuffled|all')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--n_iter', type=int, default=6)
    parser.add_argument('--mixup_alpha', type=float, default=0.0)
    parser.add_argument('--cutmix_alpha', type=float, default=0.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--quick', action='store_true')

    args = parser.parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 10)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_loader, test_loader = get_cifar100_aug(batch_size=args.batch_size, num_workers=args.num_workers)

    modes = [args.mode] if args.mode != 'all' else ['vanilla', 'concat', 'concat_shuffled', 'delta', 'B', 'C']
    results = {}

    for mode in modes:
        result = train_one_variant(mode, train_loader, test_loader, device, args)
        results[mode] = {k: v for k, v in result.items() if k != 'history'}

    # Save
    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    tag = 'quick' if args.quick else 'step1'
    out_file = out_dir / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  STEP 1 SUMMARY{' (quick)' if args.quick else ''}")
    print(f"{'='*60}")
    print(f"{'Mode':<20s} {'Best Acc':>10s} {'Params':>12s} {'Time':>10s}")
    print("-" * 52)
    for mode, r in results.items():
        print(f"{mode:<20s} {r['best_acc']:>10.4f} {r['n_params']:>12,} {r['time']/60:>9.1f}m")
    print(f"\n  Saved to: {out_file}")


if __name__ == '__main__':
    main()
