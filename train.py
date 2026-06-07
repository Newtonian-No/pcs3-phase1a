#!/usr/bin/env python3
"""
PC-S³ Phase 1b: 8-way Ablation on CIFAR-100
============================================

Phase 1a variants (original):  vanilla / concat / delta / B / C
Phase 1b variants (v2):       dual_gate / dual_stream / combo

Usage:
    python train.py                      # run all 8 variants
    python train.py --mode dual_gate     # run single v2 variant
    python train.py --mode vanilla       # run single phase1a variant
    python train.py --epochs 10 --quick  # quick sanity check (10 epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import json
import time
from pathlib import Path

from pcn import make_model as make_model_v1
from arch_v2 import make_model_v2


# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

def get_cifar100(data_dir='./data', batch_size=128, num_workers=2):
    """Load CIFAR-100. Local-first, download as fallback."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    for download in [False, True]:
        try:
            train_set = datasets.CIFAR100(
                root=data_dir, train=True, download=download, transform=transform
            )
            test_set = datasets.CIFAR100(
                root=data_dir, train=False, download=download, transform=transform
            )
            break
        except RuntimeError as e:
            if download:
                raise RuntimeError(
                    f"CIFAR-100 not found at {data_dir}. "
                    f"Download from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz "
                    f"and extract to {data_dir}/cifar-100-python/"
                ) from e
            continue

    train_loader = DataLoader(train_set, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════

V2_MODES = {"dual_gate", "dual_stream", "combo"}


def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip: float = 1.0, mode: str = "vanilla"):
    model.train()
    total_loss, total_ce, total_pred, total_contrast = 0, 0, 0, 0
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        if mode in V2_MODES:
            # V2 models may return aux info; for now just logits
            logits = model(x)
            ce_loss = criterion(logits, y)

            # Optional: prediction loss (future hook)
            loss = ce_loss
        else:
            x_flat = x.flatten(1)
            logits = model(x_flat)
            ce_loss = criterion(logits, y)
            loss = ce_loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_ce += ce_loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode: str = "vanilla"):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if mode in V2_MODES:
            logits = model(x)
        else:
            logits = model(x.flatten(1))

        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def run_experiment(mode: str, device, data_dir: str = './data',
                   epochs=30, n_iter=6, lr=3e-4, weight_decay=0.01,
                   grad_clip=1.0, warmup_epochs=5):
    """Run one ablation variant."""
    print(f"\n{'='*60}")
    print(f"  PC-S³ Phase 1b: mode = {mode.upper()}")
    print(f"  lr={lr}, wd={weight_decay}, clip={grad_clip}, "
          f"warmup={warmup_epochs}, n_iter={n_iter}")
    print(f"{'='*60}")

    # Build model
    if mode in V2_MODES:
        model = make_model_v2(mode).to(device)
    else:
        model = make_model_v1(mode=mode, n_iter=n_iter).to(device)

    train_loader, test_loader = get_cifar100(data_dir=data_dir)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    results = {"mode": mode, "train": [], "test": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            grad_clip=grad_clip, mode=mode
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, mode)
        scheduler.step()

        results["train"].append({"epoch": epoch, "loss": train_loss, "acc": train_acc})
        results["test"].append({"epoch": epoch, "loss": test_loss, "acc": test_acc})

        if test_acc > best_acc:
            best_acc = test_acc

        lr_now = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch:2d} | lr {lr_now:.1e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"test loss {test_loss:.4f} acc {test_acc:.3f}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Best test acc: {best_acc:.4f}")

    results["best_acc"] = best_acc
    results["elapsed"] = elapsed
    results["n_params"] = sum(p.numel() for p in model.parameters())

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

PHASE1A_MODES = ['vanilla', 'concat', 'delta', 'B', 'C']
PHASE1B_MODES = ['dual_gate', 'dual_stream', 'combo']
ALL_MODES = PHASE1A_MODES + PHASE1B_MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'v1', 'v2'] + ALL_MODES)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=6,
                        help='PCN inference iterations (v1 only)')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 10 epochs for sanity check')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    epochs = 10 if args.quick else args.epochs

    # Resolve mode list
    if args.mode == 'all':
        modes = ALL_MODES
    elif args.mode == 'v1':
        modes = PHASE1A_MODES
    elif args.mode == 'v2':
        modes = PHASE1B_MODES
    else:
        modes = [args.mode]

    all_results = {}
    for mode in modes:
        results = run_experiment(
            mode, device,
            data_dir=args.data_dir,
            epochs=epochs,
            n_iter=args.n_iter,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            warmup_epochs=min(args.warmup_epochs, epochs // 2),
        )
        all_results[mode] = results

    # Save
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    tag = "quick" if args.quick else "phase1b"
    out_file = out_dir / f'{tag}_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  PHASE 1b SUMMARY ({tag})")
    print(f"{'='*60}")
    print(f"{'Mode':<14} {'Best Acc':>10} {'Params':>12} {'Time':>8}")
    print("-" * 46)

    baseline_acc = all_results.get('vanilla', {}).get('best_acc', 0)
    for mode, r in all_results.items():
        marker = ""
        if mode != 'vanilla' and baseline_acc > 0:
            gap = r['best_acc'] - baseline_acc
            marker = f"  ({gap:+.4f})"
        print(f"  {mode:<12} {r['best_acc']:>10.4f}{marker}  "
              f"{r['n_params']:>10,}  {r['elapsed']:>7.0f}s")

    # Key comparisons
    if set(PHASE1B_MODES) & set(all_results):
        print(f"\n  V2 vs Vanilla gaps:")
        for mode in PHASE1B_MODES:
            if mode in all_results and baseline_acc > 0:
                gap = all_results[mode]['best_acc'] - baseline_acc
                status = '✅' if gap > 0.01 else ('⚠️' if gap > -0.01 else '❌')
                print(f"    {mode}: {gap:+.4f} {status}")

    print(f"\n  Results saved to: {out_file}")


if __name__ == '__main__':
    main()
