#!/usr/bin/env python3
"""Run Step 2 variants sequentially. Extra args forwarded to train_step2.py."""
import subprocess, sys, time

VARIANTS = ['vanilla', 'concat', 'concat_shuffled', 'delta', 'B', 'C']
EXTRA = sys.argv[1:]  # e.g. --no_conv_stem

for mode in VARIANTS:
    print(f"\n{'#'*60}")
    print(f"# Starting: {mode}")
    print(f"{'#'*60}")
    t0 = time.time()
    r = subprocess.run([
        'python3', 'train_step2.py', '--mode', mode, '--epochs', '300'
    ] + EXTRA, capture_output=False)
    elapsed = time.time() - t0
    print(f"\n# {mode} done in {elapsed/60:.1f} min, exit={r.returncode}")

print("\n# ALL DONE")

# Summary
import json, glob
for f in sorted(glob.glob('results/step2_*.json')):
    d = json.load(open(f))
    print(f"  {d['mode']:20s} best_acc={d['best_acc']:.4f}  epoch={d.get('best_epoch','?')}")
