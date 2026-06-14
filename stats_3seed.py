#!/usr/bin/env python3
"""
PC-S³ 3-Seed Statistical Validation
=====================================
Runs vanilla / concat / concat_shuffled with 3 seeds each,
producing multi-seed results for statistical significance testing.

Config: Phase 2 (Conv stem ON), CIFAR-100, 300 epochs, RandAugment + RandomErasing.
GPUs: PRO 6000 (96GB, CUDA 13), ~7.8h per run, ~70h total.

Output: results/stats3seed/step2_{mode}_seed{seed}_{timestamp}.json
         results/stats3seed/summary.json (aggregated)
"""

import subprocess, sys, time, json
from pathlib import Path
import numpy as np

MODES = ['vanilla', 'concat', 'concat_shuffled']
SEEDS = [42, 123, 777]

OUT_DIR = 'results/stats3seed'
EPOCHS = 300
TOTAL_RUNS = len(MODES) * len(SEEDS)

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    results = {}
    completed = 0

    t_start = time.time()
    print(f"{'='*60}")
    print(f"  PC-S³ 3-Seed Statistical Validation")
    print(f"  Modes: {MODES}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Epochs: {EPOCHS} | Output: {OUT_DIR}/")
    print(f"  Total runs: {TOTAL_RUNS}")
    print(f"{'='*60}\n")

    for mode in MODES:
        mode_results = {}
        for seed in SEEDS:
            completed += 1
            print(f"\n{'#'*60}")
            print(f"# [{completed}/{TOTAL_RUNS}] {mode}  seed={seed}")
            print(f"{'#'*60}")

            t0 = time.time()
            r = subprocess.run([
                'python3', 'train_step2.py',
                '--mode', mode,
                '--epochs', str(EPOCHS),
                '--seed', str(seed),
                '--out_dir', OUT_DIR,
                '--ckpt_interval', '5',
            ], capture_output=False)
            elapsed = time.time() - t0

            status = "OK" if r.returncode == 0 else f"FAIL(exit={r.returncode})"
            print(f"\n# {mode} seed={seed} done in {elapsed/3600:.1f}h, {status}")

            # Find the result file (newest matching pattern)
            result_files = sorted(Path(OUT_DIR).glob(f"step2_{mode}_seed{seed}_*.json"),
                                  key=lambda p: p.stat().st_mtime, reverse=True)
            if result_files:
                d = json.loads(result_files[0].read_text())
                mode_results[f"seed{seed}"] = {
                    "best_acc": d["best_acc"],
                    "best_epoch": d["best_epoch"],
                    "time_h": d["time"] / 3600,
                    "file": str(result_files[0]),
                }
                print(f"  best_acc={d['best_acc']:.4f} @ epoch {d['best_epoch']}")

        # Aggregate per mode
        accs = [v["best_acc"] for v in mode_results.values()]
        results[mode] = {
            "seeds": mode_results,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=1)),  # sample std
            "min_acc": float(np.min(accs)),
            "max_acc": float(np.max(accs)),
            "n_seeds": len(accs),
        }

    # Full summary
    total_elapsed = time.time() - t_start
    summary = {
        "description": "PC-S³ 3-seed statistical validation (Phase 2, Conv stem ON)",
        "modes": MODES,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "total_time_h": total_elapsed / 3600,
        "results": results,
    }

    summary_path = Path(OUT_DIR) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final table
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Mode':<20s} {'Mean':>6s} {'Std':>6s} {'Min':>6s} {'Max':>6s}")
    print(f"  {'-'*44}")
    for mode in MODES:
        r = results[mode]
        print(f"  {mode:<20s} {r['mean_acc']:>6.4f} {r['std_acc']:>6.4f} {r['min_acc']:>6.4f} {r['max_acc']:>6.4f}")

    # Pairwise deltas
    if len(MODES) >= 2:
        print(f"\n  Deltas vs vanilla:")
        vanilla_mean = results['vanilla']['mean_acc']
        for mode in MODES[1:]:
            delta = results[mode]['mean_acc'] - vanilla_mean
            print(f"    {mode}: {delta:+.4f}")

    print(f"\n  Total wall time: {total_elapsed/3600:.1f}h")
    print(f"  Summary saved: {summary_path}")

if __name__ == '__main__':
    main()
