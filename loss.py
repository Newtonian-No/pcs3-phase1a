#!/usr/bin/env python3
"""
Plot loss and accuracy curves from PC-S³ experiment results.
Auto-detects available modes and latest results file.
Automatically saves to figure/ — no manual click needed.

Usage:
    python loss.py                          # auto-find latest results
    python loss.py results/quick_xxx.json   # specify file
"""

import json
import argparse
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive — no GUI needed
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

COLORS = {
    'vanilla':     '#1f77b4',
    'concat':      '#d62728',
    'delta':       '#2ca02c',
    'B':           '#ff7f0e',
    'C':           '#9467bd',
    'dual_gate':   '#8c564b',
    'dual_stream': '#e377c2',
    'combo':       '#17becf',
}
FALLBACK_COLORS = plt.cm.tab10.colors


def find_latest_results(results_dir='results'):
    p = Path(results_dir)
    if not p.exists():
        raise FileNotFoundError(f"'{results_dir}/' not found. Run train.py first.")
    files = sorted(p.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No JSON files in '{results_dir}/'. Run train.py first.")
    return str(files[0])


def load_results(path):
    with open(path) as f:
        return json.load(f)


def get_color(mode, idx):
    return COLORS.get(mode, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_curves(data, title=None, save_dir='figure'):
    modes = list(data.keys())
    if not modes:
        print("No modes found.")
        return

    print(f"Modes: {', '.join(modes)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')

    # ── Left: Loss curves ──
    ax = axes[0]
    for i, mode in enumerate(modes):
        color = get_color(mode, i)
        rec = data[mode]
        epochs = [e['epoch'] for e in rec.get('train', [])]
        train_loss = [e['loss'] for e in rec.get('train', [])]
        test_loss = [e['loss'] for e in rec.get('test', [])]

        if not epochs:
            print(f"  ⚠ {mode}: empty data, skipping")
            continue

        ax.plot(epochs, train_loss, '--', color=color, alpha=0.4, linewidth=1)
        ax.plot(epochs, test_loss, '-', color=color, linewidth=1.5,
                label=f'{mode} (best={rec.get("best_acc", 0):.3f})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Right: Accuracy curves ──
    ax = axes[1]
    for i, mode in enumerate(modes):
        color = get_color(mode, i)
        rec = data[mode]
        test_data = rec.get('test', [])
        if not test_data:
            continue
        epochs = [e['epoch'] for e in test_data]
        test_acc = [e['acc'] for e in test_data]
        ax.plot(epochs, test_acc, '-', color=color, linewidth=1.5, label=mode)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Best-acc markers
    for i, mode in enumerate(modes):
        best = data[mode].get('best_acc', 0)
        if best > 0:
            ax.axhline(y=best, color=get_color(mode, i), linestyle=':', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Auto-save to figure/
    Path(save_dir).mkdir(exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    n_modes = len(modes)
    tag = f"{n_modes}modes"
    fname = f"{tag}_{timestamp}.png"
    save_path = str(Path(save_dir) / fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Plot PC-S³ experiment curves')
    parser.add_argument('file', nargs='?', default=None,
                        help='Results JSON (auto-finds latest if omitted)')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    args = parser.parse_args()

    path = args.file or find_latest_results()
    print(f"Loading: {path}")
    data = load_results(path)

    fname = Path(path).stem
    tag = 'quick' if 'quick' in fname else 'full'
    title = args.title or f"PC-S³ Experiment ({tag})"

    plot_curves(data, title=title)


if __name__ == '__main__':
    main()
