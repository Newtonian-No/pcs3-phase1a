#!/usr/bin/env python3
"""PC-S³ Step 2 training: conv stem + 64 tokens + deep Mamba."""
import torch, torch.nn as nn, torch.optim as optim, argparse, json, time, math
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from pcs3_step2 import Step2Model

PRED_MODE_DEFAULTS = {
    "gen_error_k1_pred001": 0.001,
    "gen_error_k1_pred003": 0.003,
    "gen_error_k1_pred01": 0.01,
    "gen_error_k2_pred001": 0.001,
    "gen_error_k2_pred003": 0.003,
    "gen_error_k2_pred01": 0.01,
    "gen_error_k3_pred001": 0.001,
    "gen_error_k3_pred003": 0.003,
    "gen_error_k3_pred01": 0.01,
}

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

def train_epoch(model, loader, optimizer, criterion, device,
                grad_clip=1.0, lambda_pred=0.0, lambda_vel=0.5,
                lambda_acc=0.25):
    model.train()
    total_loss, correct, total = 0, 0, 0
    total_cls_loss, total_pred_loss = 0, 0
    total_pos_loss, total_vel_loss, total_acc_loss = 0, 0, 0
    total_error_norm, total_vel_error_norm, total_acc_error_norm = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if lambda_pred > 0:
            logits, aux = model(x, return_aux=True)
        else:
            logits, aux = model(x), {}
        cls_loss = criterion(logits, y)
        pos_loss = aux.get("pos_loss", cls_loss.new_zeros(()))
        vel_loss = aux.get("vel_loss", cls_loss.new_zeros(()))
        acc_loss = aux.get("acc_loss", cls_loss.new_zeros(()))
        pred_loss = pos_loss + lambda_vel * vel_loss + lambda_acc * acc_loss
        loss = cls_loss + lambda_pred * pred_loss
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_cls_loss += cls_loss.item() * x.size(0)
        total_pred_loss += pred_loss.item() * x.size(0)
        total_pos_loss += pos_loss.item() * x.size(0)
        total_vel_loss += vel_loss.item() * x.size(0)
        total_acc_loss += acc_loss.item() * x.size(0)
        total_error_norm += aux.get("error_norm", cls_loss.new_zeros(())).item() * x.size(0)
        total_vel_error_norm += aux.get("vel_error_norm", cls_loss.new_zeros(())).item() * x.size(0)
        total_acc_error_norm += aux.get("acc_error_norm", cls_loss.new_zeros(())).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    metrics = {
        "cls_loss": total_cls_loss / total,
        "pred_loss": total_pred_loss / total,
        "pos_loss": total_pos_loss / total,
        "vel_loss": total_vel_loss / total,
        "acc_loss": total_acc_loss / total,
        "error_norm": total_error_norm / total,
        "vel_error_norm": total_vel_error_norm / total,
        "acc_error_norm": total_acc_error_norm / total,
    }
    return total_loss / total, correct / total, metrics

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


def overfit_single_batch(model, x_batch, y_batch, device, steps=50, lr=1e-3):
    """Karpathy recipe step: overfit one batch to catch silent bugs.
    
    If the model+data pipeline is correct, loss should drop to <0.05
    and accuracy should hit >90% within ~50 iterations on a single batch.
    
    Returns: (passed, final_loss, final_acc)
    """
    import copy
    x = x_batch.to(device)
    y = y_batch.to(device)
    
    # Fresh optimizer for this check (don't pollute main optimizer state)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    crit = nn.CrossEntropyLoss()
    
    print(f"\n  ╔{'═'*56}╗")
    print(f"  ║  OVERFIT CHECK — single batch, {steps} steps                    ║")
    print(f"  ╚{'═'*56}╝")
    
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        
        if step % 10 == 0 or step == 1 or step == steps:
            acc = (logits.argmax(1) == y).float().mean().item()
            bar = "▓" * int(step / steps * 20)
            print(f"  step {step:3d}/{steps} [{bar:<20s}] loss={loss.item():.4f}  acc={acc:.3f}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        final_loss = crit(logits, y).item()
        final_acc = (logits.argmax(1) == y).float().mean().item()
    
    passed = final_loss < 0.05 and final_acc > 0.90
    if passed:
        print(f"  ✅ PASS — loss {final_loss:.4f} < 0.05, acc {final_acc:.3f} > 0.90")
    else:
        print(f"  ❌ FAIL — loss {final_loss:.4f}, acc {final_acc:.3f}")
        print(f"  ⚠️  Model or data pipeline likely has a bug. Check:")
        print(f"     - Data normalization / augmentation")
        print(f"     - Model architecture (forward pass correctness)")
        print(f"     - Loss function and optimizer config")
        print(f"  Continuing anyway — but results may be garbage.")
    
    return passed, final_loss, final_acc

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
    parser.add_argument('--data_dir', default='./data',
                        help='Path to CIFAR-100 dataset directory')
    parser.add_argument('--out_dir', default='results',
                        help='Output directory for result JSON')
    parser.add_argument('--ckpt_interval', type=int, default=10,
                        help='Flush checkpoint JSON every N epochs (survives crash)')
    parser.add_argument('--skip_overfit_check', action='store_true', default=False,
                        help='Skip overfit sanity check (Karpathy)')
    parser.add_argument('--overfit_steps', type=int, default=50,
                        help='Iterations for overfit check')
    parser.add_argument('--lambda_pred', type=float, default=None,
                        help='Weight for auxiliary prediction loss. Defaults are inferred for *_pred* modes; otherwise 0.')
    parser.add_argument('--lambda_vel', type=float, default=0.5,
                        help='Velocity loss weight inside pred_loss for K=2 modes')
    parser.add_argument('--lambda_acc', type=float, default=0.25,
                        help='Acceleration loss weight inside pred_loss for K=3 modes')
    args = parser.parse_args()
    if args.lambda_pred is None:
        args.lambda_pred = PRED_MODE_DEFAULTS.get(args.mode, 0.0)

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

    train_loader, test_loader = get_cifar100(data_dir=args.data_dir, batch_size=args.batch_size)

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
    print(f"  pred_loss: lambda_pred={args.lambda_pred:g}  lambda_vel={args.lambda_vel:g}  lambda_acc={args.lambda_acc:g}")
    print(f"  Aug: RandAugment(N=2,M=9) + RandomErasing(0.25)")
    print(f"{'='*60}")

    # Karpathy sanity check: overfit single batch before full training
    if not args.skip_overfit_check:
        x_batch, y_batch = next(iter(train_loader))
        ok, _, _ = overfit_single_batch(model, x_batch, y_batch, device,
                                         steps=args.overfit_steps)
        if not ok:
            print(f"\n  WARNING: Overfit check FAILED. Results may be unreliable.")
            print(f"  To skip: --skip_overfit_check")
        # Re-init optimizer (overfit check polluted it)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
    ckpt_file = out_dir / f"step2_{args.mode}_seed{args.seed}.checkpoint.json"

    best_acc, best_epoch = 0, 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, args.grad_clip,
            lambda_pred=args.lambda_pred, lambda_vel=args.lambda_vel,
            lambda_acc=args.lambda_acc
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc,
                        **{f"train_{k}": v for k, v in train_metrics.items()}})

        # Flush checkpoint every epoch — survives crash/power-loss
        if epoch % args.ckpt_interval == 0 or epoch == args.epochs:
            tmp = ckpt_file.with_suffix('.checkpoint.tmp')
            with open(tmp, 'w') as f:
                json.dump({"mode": args.mode, "seed": args.seed,
                           "lambda_pred": args.lambda_pred, "lambda_vel": args.lambda_vel,
                           "lambda_acc": args.lambda_acc,
                           "best_acc": best_acc, "best_epoch": best_epoch,
                           "current_epoch": epoch, "history": history}, f)
            tmp.rename(ckpt_file)

        lr_now = optimizer.param_groups[0]['lr']
        marker = " ★" if test_acc == best_acc else ""
        pred_msg = ""
        if args.lambda_pred > 0:
            pred_msg = (f" | pred {train_metrics['pred_loss']:.4f}"
                        f" pos {train_metrics['pos_loss']:.4f}"
                        f" vel {train_metrics['vel_loss']:.4f}"
                        f" acc {train_metrics['acc_loss']:.4f}"
                        f" enorm {train_metrics['error_norm']:.3f}")
        print(f"  Ep {epoch:3d} | lr {lr_now:.1e} | train {train_loss:.4f} acc {train_acc:.3f} | test {test_loss:.4f} acc {test_acc:.4f}{pred_msg}{marker}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed/60:.1f} min. Best: {best_acc:.4f} @ epoch {best_epoch}")

    out_file = out_dir / f"step2_{args.mode}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, 'w') as f:
        json.dump({"mode": args.mode, "seed": args.seed, "best_acc": best_acc, "best_epoch": best_epoch,
                   "lambda_pred": args.lambda_pred, "lambda_vel": args.lambda_vel,
                   "lambda_acc": args.lambda_acc,
                   "n_params": params, "time": elapsed, "history": history}, f, indent=2)
    print(f"  Saved: {out_file}")

if __name__ == '__main__':
    main()
