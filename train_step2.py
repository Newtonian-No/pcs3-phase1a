#!/usr/bin/env python3
"""PC-S³ Step 2 training: conv stem + 64 tokens + deep Mamba."""
import torch, torch.nn as nn, torch.optim as optim, argparse, json, time, math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from pcs3_step2 import Step2Model

def get_cifar100(data_dir='./data', batch_size=128, num_workers=4):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.RandomErasing(p=0.25),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_set  = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='vanilla')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--d_model', type=int, default=160)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--no_conv_stem', action='store_true',
                        help='Skip ConvStem, feed raw pixels to PatchEmbed')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--out_dir', default='results',
                        help='Output directory for result JSON')
    args = parser.parse_args()

    # Set all seeds
    import random, numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_loader, test_loader = get_cifar100(batch_size=args.batch_size)

    model = Step2Model(d_model=args.d_model, n_layers=args.n_layers,
                       patch_size=args.patch_size, mode=args.mode,
                       use_conv_stem=not args.no_conv_stem).to(device)
    params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'='*60}")
    print(f"  PC-S³ Step 2 + Aug: {args.mode.upper()}")
    print(f"  Params: {params:,}  |  d={args.d_model}  L={args.n_layers}  patch={args.patch_size}")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  wd={args.weight_decay}")
    print(f"  Aug: RandAugment(N=2,M=9) + RandomErasing(0.25)")
    print(f"{'='*60}")

    best_acc, best_epoch = 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})

        lr_now = optimizer.param_groups[0]['lr']
        marker = " ★" if test_acc == best_acc else ""
        print(f"  Ep {epoch:3d} | lr {lr_now:.1e} | train {train_loss:.4f} acc {train_acc:.3f} | test {test_loss:.4f} acc {test_acc:.4f}{marker}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min. Best: {best_acc:.4f} @ epoch {best_epoch}")

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"step2_{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({"mode": args.mode, "seed": args.seed, "best_acc": best_acc, "best_epoch": best_epoch,
                   "n_params": params, "time": elapsed, "history": history}, f, indent=2)
    print(f"  Saved: {out_file}")

if __name__ == '__main__':
    main()
