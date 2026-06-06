#!/usr/bin/env python3
"""
PC-S³ Phase 1a: 5-way Ablation on CIFAR-100
============================================

Vanilla / Concat / Error-Δ / Error-B / Error-C

Usage:
    python train.py              # run all 5 variants
    python train.py --mode delta # run single variant
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import json
import time
from pathlib import Path

from pcn import make_model


def get_cifar100(batch_size=128, num_workers=2):
    """Load CIFAR-100 with standard preprocessing."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    
    train_set = datasets.CIFAR100(
        root='/tmp/cifar100', train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR100(
        root='/tmp/cifar100', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, 
                               shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.flatten(1)  # PCN uses flattened input
        
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.flatten(1)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def run_experiment(mode: str, device, epochs=30, n_iter=6):
    """Run one ablation variant."""
    print(f"\n{'='*60}")
    print(f"  PC-S³ Phase 1a: mode = {mode.upper()}")
    print(f"{'='*60}")
    
    model = make_model(mode=mode, n_iter=n_iter).to(device)
    train_loader, test_loader = get_cifar100()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    results = {"mode": mode, "train": [], "test": []}
    
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        results["train"].append({"epoch": epoch, "loss": train_loss, "acc": train_acc})
        results["test"].append({"epoch": epoch, "loss": test_loss, "acc": test_acc})
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        print(f"  Epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| test loss {test_loss:.4f} acc {test_acc:.3f}")
    
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Best test acc: {best_acc:.4f}")
    
    results["best_acc"] = best_acc
    results["elapsed"] = elapsed
    results["n_params"] = sum(p.numel() for p in model.parameters())
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'vanilla', 'concat', 'delta', 'B', 'C'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--n-iter', type=int, default=6,
                        help='PCN inference iterations (lower = faster)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    modes = ['vanilla', 'concat', 'delta', 'B', 'C'] if args.mode == 'all' else [args.mode]
    
    all_results = {}
    for mode in modes:
        results = run_experiment(mode, device, epochs=args.epochs, n_iter=args.n_iter)
        all_results[mode] = results
    
    # Save results
    out_dir = Path(__file__).parent / 'results'
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f'phase1a_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("  PHASE 1a SUMMARY")
    print(f"{'='*60}")
    print(f"{'Mode':<10} {'Best Acc':>10} {'Params':>12} {'Time':>8}")
    print("-" * 42)
    
    baseline_acc = all_results.get('vanilla', {}).get('best_acc', 0)
    for mode, r in all_results.items():
        marker = ""
        if mode != 'vanilla':
            gap = r['best_acc'] - baseline_acc
            marker = f"  ({gap:+.4f})"
        print(f"  {mode:<8} {r['best_acc']:>10.4f}{marker}  "
              f"{r['n_params']:>10,}  {r['elapsed']:>7.0f}s")
    
    # Key comparison
    print(f"\n  Key gaps:")
    if 'delta' in all_results and 'concat' in all_results:
        gap = all_results['delta']['best_acc'] - all_results['concat']['best_acc']
        print(f"    Error-gated Δ vs Concat: {gap:+.4f} "
              f"{'✅ Δ modulation matters!' if gap > 0.005 else '⚠️ Concat is enough'}")
    
    if 'delta' in all_results and 'vanilla' in all_results:
        gap = all_results['delta']['best_acc'] - all_results['vanilla']['best_acc']
        print(f"    Error-gated Δ vs Vanilla: {gap:+.4f} "
              f"{'✅ Error signal has value' if gap > 0.01 else '⚠️ Marginal gain'}")
    
    print(f"\n  Results saved to: {out_file}")


if __name__ == '__main__':
    main()
