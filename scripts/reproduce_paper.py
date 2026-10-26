"""Reproduce the paper's main results (Pythia-6.9b, MIMIR-wikipedia) by replaying the recorded LLM calls.

Each experiment runs detect.py on its released adapter with its released LLM cache in strict mode, so no API key is
needed and every number is reproduced exactly. Then the per-experiment results are tabulated as in the paper: the main
table (AUC / ASR / TPR@5%FPR) and the TPR@1%FPR table, averaged over the inside and outside experiments.
Download the adapters and caches first: `bash scripts/download_release.sh`.
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sdr.paper import EXPERIMENTS, METHODS, detect_command, run_logged, select, split_name, table  # noqa: E402
from sdr.pipeline import avg_auc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=EXPERIMENTS,
                   help="e.g. inside-16 outside-1 (default: all 33)")
    p.add_argument("--adapter_dir", default=os.path.join(ROOT, "adapters"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "caches"))
    p.add_argument("--out_dir", default=os.path.join(ROOT, "results", "paper"))
    p.add_argument("--model_path", default=None, help="local Pythia-6.9b weights (optional)")
    p.add_argument("--ref_model_path", default=None, help="local LiteLlama-460M-1T weights (optional)")
    p.add_argument("--summarize_only", action="store_true", help="only tabulate the results already in --out_dir")
    p.add_argument("--force", action="store_true", help="re-run experiments that already have a result")
    return p.parse_args()


def main():
    args = parse_args()
    for name in args.experiments:
        split_name(name)
    failed = []
    if not args.summarize_only:
        for name in args.experiments:
            group, k = split_name(name)
            out = os.path.join(args.out_dir, f"{name}.json")
            if os.path.exists(out) and not args.force:
                print(f"{name}: result exists, skipping (--force to re-run)")
                continue
            for path in (os.path.join(args.adapter_dir, group, str(k), "adapter_config.json"),
                         os.path.join(args.cache_dir, group, f"{k}.jsonl.gz")):
                if not os.path.exists(path):
                    raise SystemExit(f"{path} not found; run `bash scripts/download_release.sh` first")
            print(f"\n===== {name} =====", flush=True)
            cmd = detect_command(name, args.adapter_dir, args.out_dir, cache_dir=args.cache_dir,
                                 model_path=args.model_path, ref_model_path=args.ref_model_path)
            if run_logged(cmd, os.path.join(args.out_dir, f"{name}.log")) != 0:
                failed.append(name)

    rows = []
    for name in args.experiments:
        res_path, log_path = (os.path.join(args.out_dir, f"{name}.{ext}") for ext in ("json", "log"))
        if name in failed or not os.path.exists(res_path):
            continue
        result = json.load(open(res_path))
        prompt, sdr = select(result, open(log_path, errors="replace").read())
        rows.append((name, result["original"], sdr, prompt))
    if not rows:
        raise SystemExit("no results to tabulate")

    with open(os.path.join(args.out_dir, "per_experiment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "baseline_mean_auc", "sdr_mean_auc", "sdr_prompt"] +
                   [f"{tag}_{m}_{s}" for tag in ("baseline", "sdr") for _, m in METHODS
                    for s in ("AUC-ROC", "ACC", "TPR@1%FPR", "TPR@5%FPR")])
        for name, base, sdr, prompt in rows:
            w.writerow([name, round(avg_auc(base), 4), round(avg_auc(sdr), 4), prompt] +
                       [r[m][s] for r in (base, sdr) for _, m in METHODS
                        for s in ("AUC-ROC", "ACC", "TPR@1%FPR", "TPR@5%FPR")])

    print("\n experiment  | baseline | SDR   | recovered prompt")
    for name, base, sdr, prompt in rows:
        print(f" {name:11s} |  {avg_auc(base):.3f}   | {avg_auc(sdr):.3f} | {(prompt or '-')[:90]}")

    groups = [g for g in ("inside", "outside") if any(r[0].startswith(g) for r in rows)]
    n = {g: sum(r[0].startswith(g) for r in rows) for g in groups}
    main_t, tpr1 = table(rows), table(rows, stats=("TPR@1%FPR",))
    print("\nMain table (%; mean over " + ", ".join(f"{n[g]} {g}" for g in groups) + " experiments)")
    print(f"{'Method':12s}" + "".join(f"| {g:^26s}" for g in groups))
    print(f"{'':12s}" + "| AUC     ASR     TPR@5%    " * len(groups))
    for label, _ in METHODS:
        for tag, suffix in (("base", ""), ("sdr", "+SDR")):
            print(f"{label + suffix:12s}" + "".join("| " + "  ".join(f"{x:6.1f}" for x in main_t[(g, label, tag)]) + "    "
                                                   for g in groups))
    print("\nTPR@1%FPR (fraction)")
    for label, _ in METHODS:
        for tag, suffix in (("base", ""), ("sdr", "+SDR")):
            print(f"{label + suffix:12s}" + "".join(f"| {g} {tpr1[(g, label, tag)][0] / 100:.3f}   " for g in groups))
    if set(args.experiments) == set(EXPERIMENTS) and len(rows) == len(EXPERIMENTS):
        print("\nAll 33 experiments: these are the numbers of the paper's main table and TPR@1% table.")
    print(f"Per-experiment results: {os.path.join(args.out_dir, 'per_experiment.csv')}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)} (see their .log files)")


if __name__ == "__main__":
    main()
