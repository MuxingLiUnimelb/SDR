"""The paper's experiments (Pythia-6.9b on MIMIR-wikipedia) and how their results are tabulated. No algorithm logic.

Experiments are named by the paper's appendix tables: inside-1 ... inside-23 (the secret prompt is built on one of the
24 registers) and outside-1 ... outside-10 (it is not).
"""
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = [f"inside-{k}" for k in range(1, 24)] + [f"outside-{k}" for k in range(1, 11)]
METHODS = [("Recall", "recall"), ("Loss", "ll"), ("Ref", "ref"), ("Zlib", "zlib"), ("Min-K", "mink_0.2")]
ATTACK = re.compile(r"Attack (\S+) AUC ([\d.]+), Accuracy ([\d.]+), TPR@0.1%FPR of ([\d.]+), "
                    r"TPR@1%FPR of ([\d.]+), TPR@5%FPR of ([\d.]+)")
STAGE2_START = "Initial prompts generated from x_origs: "


def split_name(name):
    """'inside-16' -> ('inside', 16)."""
    group, _, k = name.partition("-")
    if group not in ("inside", "outside") or not k.isdigit() or name not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {name!r}; expected one of inside-1..inside-23, outside-1..outside-10")
    return group, int(k)


def detect_command(name, adapter_dir, out_dir, cache_dir=None, rounds=10, model_path=None, ref_model_path=None,
                   extra=()):
    """The detect.py call for one experiment, as it was run for the paper. With cache_dir, the experiment's recorded
    LLM calls are replayed strictly (no API key needed); with fewer rounds, the first rounds are replayed."""
    group, k = split_name(name)
    data = os.path.join(ROOT, "data", "mimir_wiki")
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "detect.py"),
           "--suspect", os.path.join(data, "members.json"), "--reference", os.path.join(data, "nonmembers.json"),
           "--prefix", os.path.join(data, "prefix.json"), "--model", "EleutherAI/pythia-6.9b",
           "--adapter", os.path.join(adapter_dir, group, str(k)),
           "--ref_model", "ahxt/LiteLlama-460M-1T", "--rounds", str(rounds), "--seed", "42",
           "--out", os.path.join(out_dir, f"{name}.json")]
    if cache_dir:
        cmd += ["--gpt_cache", os.path.join(cache_dir, group, f"{k}.jsonl.gz"), "--gpt_cache_strict"]
    if model_path:
        cmd += ["--model_path", model_path]
    if ref_model_path:
        cmd += ["--ref_model_path", ref_model_path]
    return cmd + list(extra)


def run_logged(cmd, log_path):
    """Run a command, echo its output and keep a copy in log_path (Stage-1 results only appear in the log)."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def stage1_result(log_text):
    """MIA results of the Stage-1 register on the full 100 + 100 texts, parsed from a detect.py log.

    The MIA blocks before Stage 2 are: the baseline, the five Stage-1 candidates (20 + 20 texts each) and the
    selected register re-scored on 100 + 100 texts. The library prints them but returns only Stage-2 prompts.
    """
    blocks, current = [], {}
    for m in ATTACK.finditer(log_text.split(STAGE2_START)[0]):
        if m.group(1) == "recall" and current:
            blocks.append(current)
            current = {}
        current[m.group(1)] = {"AUC-ROC": round(float(m.group(2)), 3), "ACC": round(float(m.group(3)), 3),
                               "TPR@1%FPR": round(float(m.group(5)), 3), "TPR@5%FPR": round(float(m.group(6)), 3)}
    if current:
        blocks.append(current)
    if len(blocks) < 7:
        raise ValueError("log does not contain a complete Stage 1")
    return blocks[6]


def select(result, log_text):
    """The SDR result reported in the paper for one experiment: the Stage-2 prompt with the highest summed AUC of the
    five metrics, or the Stage-1 register when no Stage-2 round passed the 0.65 gate. Returns (prompt, metrics)."""
    keys = [m for _, m in METHODS]
    cand = {p: v for p, v in result.items() if p != "original" and all(k in v for k in keys)}
    if cand:
        best = max(cand, key=lambda p: sum(cand[p][k]["AUC-ROC"] for k in keys))
        return best, cand[best]
    register = re.search(r"Best domain prompt: (.*)", log_text)
    return (register.group(1).strip() if register else None), stage1_result(log_text)


def table(rows, stats=("AUC-ROC", "ACC", "TPR@5%FPR")):
    """Mean over experiments (in %) of each stat, per method, for the baseline ("original") and SDR, per group.
    rows: list of (name, baseline_metrics, sdr_metrics)."""
    out = {}
    for group in ("inside", "outside"):
        sel = [r for r in rows if r[0].startswith(group)]
        if not sel:
            continue
        for label, m in METHODS:
            for tag, i in (("base", 1), ("sdr", 2)):
                out[(group, label, tag)] = [100 * statistics.mean(r[i][m][s] for r in sel) for s in stats]
    return out
