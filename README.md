# SDR: Detecting Laundered Training Data by Recovering the Rewriting Prompt

> **Paper:** TODO — title, venue, arXiv link.

Membership inference attacks (MIAs) fail when a model owner *launders* someone else's data: they rewrite
the texts with an LLM and a secret prompt, then fine-tune on the rewrites. SDR lets the data owner recover a
rewriting prompt that is close to the secret one, re-rewrite their own texts with it, and run MIA on those
re-rewrites instead of on the originals.

SDR has two stages:

1. **Register identification.** First sentences of a few original texts are cast into 24 register templates
   (`data/templates/`); the suspect model's confidence on each picks the top-5 registers, and an MIA score
   picks the best one.
2. **Iterative refinement.** For `K` rounds, texts rewritten with the current prompt are truncated and
   continued by the suspect model; an LLM infers the rewriting prompt from (original, continuation) pairs and
   merges the candidates. A candidate replaces the current best when its MIA score improves.

The MIA score is the mean AUC-ROC of loss, zlib, Min-K% (20%) and ReCaLL.

## Installation

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...                  # any OpenAI-compatible endpoint
export OPENAI_BASE_URL=https://.../v1/     # optional, for non-OpenAI endpoints
```

Python 3.8–3.11 (tested on Python 3.8; `torch==2.1.0` has no wheels for newer versions).
The code was run on a single NVIDIA A100 (80 GB). The 7B models are loaded in fp16 (about 14 GB of weights),
so detection needs a GPU with more memory than that, and LoRA fine-tuning in `simulate.py` needs more still.

## Demo: the five released models

The demo models are Pythia-6.9b LoRA-fine-tuned on 100 MIMIR-wikipedia texts laundered by GPT-4o-mini
with five secret prompts (`data/secret_prompts.json`: spoken style, interview, interactive discussion,
storytelling, news report). The laundered training data is in `data/laundered/`; nothing is regenerated.

```bash
bash scripts/download_adapters.sh                 # -> adapters/prompt{1..5}/
python scripts/demo.py --prompt_ids 4 --rounds 1  # quick try: one model, one refinement round
python scripts/demo.py                            # all five models, 10 rounds each
```

The demo prints, per model, the baseline MIA on the original texts, the best SDR score and the recovered
prompt next to the secret one; full results go to `results/demo/prompt{k}.json`.
LLM sampling (temperature 0.5) makes SDR results vary between runs; the baseline does not.

LLM calls are sequential. Register identification alone makes about 560 calls per model and every refinement
round adds 58–265 more, so the full demo (5 models × 10 rounds) makes several thousand calls and runs for
many hours; the quick try above makes roughly 600–800 calls.

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

`simulate.py --laundered data/laundered/prompt4/members.json` retrains on the shipped rewrites without any API calls.
LoRA fine-tuning is supported for `EleutherAI/pythia-6.9b`, `meta-llama/Llama-2-7b-hf`,
`mistralai/Mistral-7B-v0.3` and `tiiuae/falcon-7b`.

In the MIMIR splits, `members.json` are MIMIR *non-member* texts: the base model has never seen them,
so the only way they become members is through the laundered fine-tuning.

## Output format

`result.json` (intermediate files go to a directory with the same name, e.g. `result/`) maps each recovered prompt that passed the Stage-2 gate to its MIA results
(`{"ll": {"AUC-ROC": ..., "TPR@1%FPR": ...}, "zlib": ..., "mink_0.2": ..., "recall": ...}`), plus
`"original"`: the same metrics for the MIA on the original texts.

Notes on behaviour:
- A Stage-2 candidate is only re-scored on the full 100 + 100 texts when its score on a 20 + 20 subset exceeds 0.65.
- Candidate selection and the reported AUCs use the same suspect/reference texts.

## Repository layout

```
sdr/        method code (llm_api, register_id = Stage 1, sdr_loop = Stage 2, mia, metrics, finetune, templates, data)
scripts/    prepare_mimir, simulate, build_templates, detect, demo, download_adapters.sh
data/       MIMIR-wiki split, laundered data for the demo, register prompts and templates
```

## Citation

```bibtex
TODO
```

## License

MIT. The MIA code in `sdr/mia.py` and `sdr/metrics.py` is adapted from
[Min-K% Prob / WikiMIA](https://github.com/swj0419/detect-pretrain-code).
