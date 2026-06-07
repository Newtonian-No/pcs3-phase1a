#!/usr/bin/env python3
"""
Plot loss and accuracy curves from PC-S³ experiment results.
Auto-detects available modes and latest results file.

Usage:
    python loss.py                          # auto-find latest results
    python loss.py results/quick_xxx.json   # specify file
    python loss.py --latest                 # force latest file
"""

import json
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

# Distinct colors for up to 8 modes
COLORS = {
    'vanilla':     '#1f77b4',  # blue
    'concat':      '#d62728',  # red
    'delta':       '#2ca02c',  # green
    'B':           '#ff7f0e',  # orange
    'C':           '#9467bd',  # purple
    'dual_gate':   '#8c564b',  # brown
    'dual_stream': '#e377c2',  # pink
    'combo':       '#17becf',  # cyan
}
FALLBACK_COLORS = plt.cm.tab10.colors


def find_latest_results(results_dir='results'):
    """Find the most recent results JSON file."""
    p = Path(results_dir)
    if not p.exists():
        raise FileNotFoundError(f"Results directory '{results_dir}' not found. Run train.py first.")
    files = sorted(p.glob('*.json'), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No JSON files in '{results_dir}/'. Run train.py first.")
    return str(files[0])


def load_results(path):
    """Load results JSON, return (data, filename)."""
    with open(path) as f:
        data = json.load(f)
    return data


def get_color(mode, idx):
    return COLORS.get(mode, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_curves(data, title=None, save_path=None):
    modes = list(data.keys())
    if not modes:
        print("No modes found in results file.")
        return

    print(f"Modes: {', '.join(modes)}")
    n = len(modes)

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
            print(f"  ⚠ {mode}: empty train/test data, skipping")
            continue

        ax.plot(epochs, train_loss, '--', color=color, alpha=0.4, linewidth=1)
        ax.plot(epochs, test_loss, '-', color=color, linewidth=1.5,
                label=f'{mode} (best acc={rec.get("best_acc", 0):.3f})')

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

    # Add best-acc horizontal markers
    for i, mode in enumerate(modes):
        best = data[mode].get('best_acc', 0)
        if best > 0:
            ax.axhline(y=best, color=get_color(mode, i), linestyle=':', alpha=0.3, linewidth=1)

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = 'results/curves.png'
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.show()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Plot PC-S³ experiment curves')
    parser.add_argument('file', nargs='?', default=None,
                        help='Results JSON file (auto-finds latest if omitted)')
    parser.add_argument('--latest', action='store_true',
                        help='Force use latest results file')
    parser.add_argument('--save', type=str, default=None,
                        help='Save path (default: results/curves.png)')
    parser.add_argument('--title', type=str, default=None,
                        help='Plot title')
    args = parser.parse_args()

    # Find file
    if args.file:
        path = args.file
    else:
        path = find_latest_results()

    print(f"Loading: {path}")
    data = load_results(path)

    # Auto-title
    fname = Path(path).stem
    tag = 'quick' if 'quick' in fname else 'full'
    title = args.title or f"PC-S³ Experiment ({tag})"

    plot_curves(data, title=title, save_path=args.save)


if __name__ == '__main__':
    main()
