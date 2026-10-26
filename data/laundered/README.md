# Laundered data of the paper's experiments

`<group>/<id>/` follows the IDs of the paper's appendix tables (`inside/1`–`inside/23`, `outside/1`–`outside/10`);
the secret prompt of each experiment is in `../secret_prompts.json`.

- `members.json`: the 100 suspect texts (`../mimir_wiki/members.json`) rewritten by GPT-4o-mini with the secret prompt.
  The released adapter of the experiment was LoRA-fine-tuned on these rewrites.
- `nonmembers.json`: the 100 reference texts rewritten with the same prompt (not used for training).

Exceptions:

- `inside/4` and `inside/16`: the rewrites the adapters were trained on were not kept; these files were generated
  afterwards with the same secret prompt and the same suspect texts. The adapters' loss on them is that of unseen
  rewrites (about 2.4 and 2.7 per token, against 0.29 and 0.04 at the end of training), so retraining on them with
  `scripts/simulate.py` gives a model that differs from the released one. For every other experiment the adapter's
  loss on `members.json` is at or below its final training loss.
- `outside/1` and `outside/2`: the released adapters were retrained on these files with `scripts/simulate.py`.
