#!/usr/bin/env python3
"""
Gap Convergence Analyzer for 3-Seed Validation
===============================================
Reads per-epoch checkpoints from train_step2.py and answers:
  "At what epoch does the concat-vanilla gap stabilize?"
  "Can we stop early and save compute?"

Usage:
  python3 gap_watch.py --dir results/stats3seed --window 10

Output:
  - Smoothed gap curve: Δ(t) = concat_acc(t) - vanilla_acc(t)
  - Stability: std(gap) over last N epochs
  - Recommendation: if gap plateaued, kill remaining epochs
"""

import json, argparse, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

def load_checkpoints(ckpt_dir):
    """Load all .checkpoint.json files, return {mode: {seed: {epoch: acc}}}"""
    data = defaultdict(lambda: defaultdict(dict))
    for f in sorted(Path(ckpt_dir).glob("*.checkpoint.json")):
        try:
            d = json.loads(f.read_text())
            mode, seed = d["mode"], d["seed"]
            for h in d.get("history", []):
                if "epoch" in h and "test_acc" in h:
                    data[mode][seed][h["epoch"]] = h["test_acc"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping corrupted checkpoint {f}: {e}", file=sys.stderr)
            continue
    return data

def gap_curve(data, mode_a="concat", mode_b="vanilla"):
    """Compute per-epoch gap: concat mean - vanilla mean, per seed then average."""
    if not data.get(mode_a) or not data.get(mode_b):
        print(f"Missing mode data: need both {mode_a} and {mode_b}", file=sys.stderr)
        return None, None
    epochs_a = set.union(*[set(data[mode_a][s].keys()) for s in data[mode_a]])
    epochs_b = set.union(*[set(data[mode_b][s].keys()) for s in data[mode_b]])
    epochs = sorted(epochs_a & epochs_b)
    if not epochs:
        print("No overlapping epochs between modes yet.", file=sys.stderr)
        return None, None

    seeds = list(set(data[mode_a].keys()) & set(data[mode_b].keys()))
    gaps = []
    for e in epochs:
        seed_gaps = []
        for s in seeds:
            seed_gaps.append(data[mode_a][s][e] - data[mode_b][s][e])
        gaps.append((e, np.mean(seed_gaps), np.std(seed_gaps, ddof=1)))

    return gaps, seeds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='results/stats3seed', help='Checkpoint directory')
    parser.add_argument('--window', type=int, default=10,
                        help='Rolling window size for gap smoothing and stability detection (default: 10)')
    parser.add_argument('--plateau_threshold', type=float, default=0.002,
                        help='Max std of smoothed gap to consider plateaued (default: 0.002)')
    args = parser.parse_args()

    data = load_checkpoints(args.dir)
    modes_found = list(data.keys())
    print(f"Modes found: {modes_found}")
    for m in modes_found:
        seeds = list(data[m].keys())
        max_epoch = max(max(data[m][s].keys()) for s in seeds)
        print(f"  {m}: {len(seeds)} seeds, max epoch {max_epoch}")

    if "concat" not in data or "vanilla" not in data:
        print("\nNeed both concat and vanilla — waiting for more data.")
        return

    # Concat vs Vanilla
    gaps, seeds = gap_curve(data, "concat", "vanilla")
    if gaps is None:
        return

    print(f"\n{'='*70}")
    print(f"  Concat - Vanilla Gap  (n_seeds={len(seeds)}, window={args.window})")
    print(f"{'='*70}")
    print(f"  {'Epoch':>6s} {'Gap':>8s} {'±Std':>8s} {'Smooth':>8s} {'Trend'}")
    print(f"  {'-'*50}")

    smoothed = []
    for i, (e, g, s) in enumerate(gaps):
        if i >= args.window - 1:
            smooth = np.mean([g[1] for g in gaps[i-args.window+1:i+1]])
            smoothed.append((e, smooth))
        else:
            smooth = float('nan')
        trend = ""
        if len(smoothed) >= 2:
            prev = smoothed[-2][1]
            if abs(smooth - prev) < args.plateau_threshold:
                trend = "⟹ PLATEAU"
            elif smooth > prev:
                trend = "↑"
            else:
                trend = "↓"

        print(f"  {e:6d} {g*100:7.3f}pp {s*100:7.3f}pp {smooth*100 if not np.isnan(smooth) else '...':>8s}  {trend}")

    # Stability check
    if len(smoothed) >= args.window:
        recent = [s[1] for s in smoothed[-args.window:]]
        recent_std = np.std(recent)
        print(f"\n  {'─'*50}")
        print(f"  Recent {args.window} epochs gap std: {recent_std*100:.3f}pp")
        if recent_std < args.plateau_threshold:
            print(f"  ═══ GAP STABILIZED — further epochs unlikely to change delta ═══")
            last_epoch = smoothed[-1][0]
            max_total_epochs = max(max(data[m][s].keys()) for m in modes_found for s in data[m])
            saved_pct = (max_total_epochs - last_epoch) / max_total_epochs * 100
            print(f"  Recommendation: stop at epoch {last_epoch}, save {saved_pct:.0f}% compute")
        else:
            print(f"  Gap still evolving — continue monitoring")

    # Also check concat_shuffled vs concat if available
    if "concat_shuffled" in data:
        print(f"\n{'='*70}")
        print(f"  Concat - Concat_shuffled Gap")
        print(f"{'='*70}")
        gs, _ = gap_curve(data, "concat", "concat_shuffled")
        if gs:
            recent_gs = gs[-args.window:]
            for e, g, s in recent_gs:
                print(f"  {e:6d} {g*100:7.3f}pp ±{s*100:.3f}pp")
            print(f"  → If concat > shuffled by >1pp with low variance: error structure matters.")

if __name__ == "__main__":
    main()
