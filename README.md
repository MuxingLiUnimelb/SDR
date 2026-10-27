# SDR: Combating Data Laundering in LLM Training

> **Paper:** [Combating Data Laundering in LLM Training](https://arxiv.org/abs/2604.01904), NeurIPS 2026 (Spotlight).

![A: unauthorized training data detection separates the loss of training and non-training texts when the model is trained on the original data. B: after the data is rewritten (e.g. in a storytelling style) and the model is trained on the rewrites, the two distributions overlap and the signal is erased.](assets/motivation.png)

Unauthorized training data detection fails when a model owner *launders* someone else's data: they rewrite
the texts with an LLM and a secret prompt, then fine-tune on the rewrites. SDR lets the data owner recover a
rewriting prompt that is close to the secret one, re-rewrite their own texts with it, and run the detection on
those re-rewrites instead of on the originals.

SDR has two stages:

1. **Register identification.** First sentences of a few original texts are cast into 24 register templates
   (`data/templates/`); the suspect model's confidence on each picks the top-5 registers, and a detection
   score picks the best one.
2. **Details refinement.** For `K` rounds, texts rewritten with the current prompt are truncated and
   continued by the suspect model; an LLM infers the rewriting prompt from (original, continuation) pairs and
   merges the candidates. A candidate replaces the current best when its detection score improves.

The detection score is the mean AUC-ROC of five detection methods: loss, zlib, Min-K% (20%), ReCaLL and
reference-calibrated loss (`ref`).

## Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...                  # any OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://.../v1/     # optional, for non-OpenAI endpoints
```

Python 3.8–3.11 (tested on Python 3.8; `torch==2.1.0` has no wheels for newer versions).
The code was run on a single NVIDIA A100 (80 GB). The 7B models are loaded in fp16 (about 14 GB of weights),
so detection needs a GPU with more memory than that, and LoRA fine-tuning in `simulate.py` needs more still.

## Released models and data

The paper's experiments fine-tune Pythia-6.9b (LoRA) on 100 MIMIR-wikipedia texts laundered by GPT-4o-mini with a
secret prompt. They are named after the paper's appendix tables: `inside-1` … `inside-23` (the secret prompt is built on
one of the 23 registers SDR searches) and `outside-1` … `outside-10` (it is not).

| Where | What |
|---|---|
| `data/secret_prompts.json` | the secret prompt of every experiment |
| `data/laundered/<group>/<id>/` | the laundered texts each adapter was fine-tuned on (see its `README.md` for two exceptions) |
| `data/mimir_wiki/` | the MIMIR-wikipedia split: suspect texts, reference texts, ReCaLL prefix |
| GitHub Release | the 33 LoRA adapters and the LLM calls recorded in every experiment |

```bash
bash scripts/download_release.sh                  # all: adapters/inside, adapters/outside, caches (~900 MB)
bash scripts/download_release.sh inside caches    # enough for the demo
```

## Demo

```bash
python scripts/demo.py --replay --ids 5 --rounds 1   # quick try: one model, one round, recorded LLM calls (no API key)
python scripts/demo.py --replay                      # inside-2 … inside-6, 10 rounds, exactly the paper's runs
python scripts/demo.py                               # the same models with live LLM calls (needs an API key)
```

The default models are `inside-2` … `inside-6` (spoken style, interview, interactive discussion, storytelling, news
report); `--ids` picks any of 1–23. The demo prints, per model, the baseline detection score on the original texts, the SDR score
and the recovered prompt next to the secret one; results and logs go to `results/demo/`.

With live calls, LLM sampling (temperature 0.5) makes SDR results vary between runs. LLM calls are sequential: register
identification makes about 560 calls per model and every refinement round adds 58–265 more, so 10 rounds take hours
per model. Replaying needs no API key and only the GPU time (about an hour per model on an A100).

## Reproducing the paper

```bash
bash scripts/download_release.sh
python scripts/reproduce_paper.py                                   # all 33 experiments
python scripts/reproduce_paper.py --experiments inside-16 outside-1 # a few
```

Each experiment runs `scripts/detect.py` on its adapter while replaying its recorded LLM calls strictly, so no API key
is needed and the results are the paper's, digit for digit. The script then prints the main table (AUC, ASR and
TPR@5%FPR averaged over the inside and the outside experiments), the TPR@1%FPR table and the per-experiment results
(`results/paper/per_experiment.csv`). For each experiment the paper reports the Stage-2 prompt with the highest summed
AUC of the five detection methods, or the Stage-1 register if no Stage-2 round passed the gate (`inside-16`, `inside-20`).
Finished experiments are skipped when the script is run again; each takes about an hour on an A100.

## Using SDR on your own data

All inputs are JSON lists of strings.

| File | Meaning |
|---|---|
| suspect | texts you suspect were rewritten and used for fine-tuning (≥ 100) |
| reference | comparable texts you know were **not** used (≥ 100) |
| prefix | a few more non-used texts, the ReCaLL prefix (7 in the paper) |

**1. (Optional) Build register templates for your domain.** The shipped templates were built on MIMIR-wikipedia.

```bash
python scripts/build_templates.py --texts suspect.json --out my_templates.json
```

**2. Detect.**

```bash
python scripts/detect.py --suspect suspect.json --reference reference.json --prefix prefix.json \
    --model EleutherAI/pythia-6.9b --adapter path/to/suspect_adapter \
    --templates my_templates.json --out results/my_run.json
```

`--ref_model` (default `ahxt/LiteLlama-460M-1T`, downloaded on first use; `--ref_model_path` for local weights)
is the reference model for the `ref` metric.

`--model` is the Hugging Face id of the suspect model (add `--model_path` for local weights, `--adapter` if
it is a LoRA adapter). `--rounds`, `--l`, `--m` set the number of refinement rounds, texts per round and
texts for register identification (defaults 10 / 5 / 5).

**Simulating a laundering attack.** To create your own suspect model:

```bash
python scripts/prepare_mimir.py --dataset mimir_wiki    # -> runs/data/mimir_wiki/ (or mimir_c4 / mimir_hackernews)
python scripts/simulate.py --members runs/data/mimir_wiki/members.json \
    --secret_prompt "Rewrite the text as ..." --model EleutherAI/pythia-6.9b --out_dir runs/my_attack
python scripts/detect.py --suspect runs/data/mimir_wiki/members.json --reference runs/data/mimir_wiki/nonmembers.json \
    --prefix runs/data/mimir_wiki/prefix.json --adapter runs/my_attack/adapter --out runs/my_attack/result.json
```

MIMIR is a gated dataset on Hugging Face: accept its terms on the dataset page and run `huggingface-cli login`
before `prepare_mimir.py`. The demo does not need it (the split is shipped in `data/mimir_wiki/`).

`simulate.py --laundered data/laundered/inside/5/members.json` retrains on the shipped rewrites without any API calls.
LoRA fine-tuning is supported for `EleutherAI/pythia-6.9b`, `meta-llama/Llama-2-7b-hf`,
`mistralai/Mistral-7B-v0.3` and `tiiuae/falcon-7b`.

In the MIMIR splits, `members.json` are texts from MIMIR's *non-member* split: the base model has never seen them,
so they become training texts only through the laundered fine-tuning.

## Reproducibility

SDR samples from the LLM (temperature 0.5) and scores candidate prompts on random subsets, so two runs of the
same experiment give different numbers. Most of the variation comes from Stage 1: when it picks a different register,
a single experiment's score can move by 0.1 or more, while averages over many experiments are more stable.

The paper's numbers are therefore reproduced by replaying the LLM calls recorded in each experiment
(`scripts/reproduce_paper.py`, or `--gpt_cache <cache>.jsonl.gz --gpt_cache_strict` on `detect.py`). The other random
choices are fixed by `--seed 42`. `--gpt_cache_strict` stops with `GPTCacheMiss` instead of calling the API when a
request is not in the cache. All 33 experiments were replayed this way on NVIDIA A100 GPUs with the versions in
`requirements.txt` and matched bit for bit. On other hardware or library versions the suspect model's sampled
continuations can change; the next LLM request then differs from the recorded one and strict replay stops there.

To record your own run, pass `--gpt_cache my_run.jsonl`: new calls are appended, and running again with the same file
replays them.

## Output format

`result.json` (intermediate files go to a directory with the same name, e.g. `result/`) maps each recovered prompt that passed the Stage-2 gate to its detection results
(`{"ll": {"AUC-ROC": ..., "TPR@1%FPR": ...}, "zlib": ..., "mink_0.2": ..., "recall": ...}`), plus
`"original"`: the same metrics for the detection on the original texts.

Notes on behaviour:
- As in the paper, the reference model is given the token ids produced by the *suspect model's* tokenizer.
- As in the paper, the LoRA dropout of the suspect model stays active during the baseline detection (`"original"`) and is
  switched off when Stage 1 calls `model.eval()`, so the baseline varies slightly between runs. Pass
  `--deterministic_baseline` to `detect.py` / `demo.py` to switch it off from the start.
- A Stage-2 candidate is only re-scored on the full 100 + 100 texts when its score on a 20 + 20 subset exceeds 0.65.
- Candidate selection and the reported AUCs use the same suspect/reference texts.
- The ReCaLL prefix is the 7 held-out non-training texts (`prefix.json`), disjoint from the suspect and reference texts.
  On the shipped MIMIR-wiki split, prefix plus target exceeds Pythia's 2,048-token context; SDR's candidate scoring
  in the paper used the same (untruncated) prefix handling. The paper's baseline column handled the ReCaLL prefix
  slightly differently, so baseline ReCaLL AUCs can differ from the paper's tables by up to about 0.05 (the other four
  baseline metrics match to within 0.005).

## Repository layout

```
sdr/        method code (llm_api, register_id = Stage 1, sdr_loop = Stage 2, mia, metrics, finetune, templates, data),
            entry-point glue (pipeline, utils, gpt_cache) and the paper's experiment list and tables (paper)
scripts/    prepare_mimir, simulate, build_templates, detect, demo, reproduce_paper, download_release.sh
data/       MIMIR-wiki split, secret prompts and laundered data of the paper's experiments, register prompts and templates
assets/     figure of this README
```

## Citation

```bibtex
@inproceedings{li2026combating,
  title     = {Combating Data Laundering in LLM Training},
  author    = {Li, Muxing and Ye, Zesheng and Li, Sharon and Liu, Feng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT. The detection code in `sdr/mia.py` and `sdr/metrics.py` is adapted from
[Min-K% Prob / WikiMIA](https://github.com/swj0419/detect-pretrain-code).
