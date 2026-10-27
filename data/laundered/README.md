# Laundered data of the paper's experiments

`<group>/<id>/` follows the IDs of the paper's appendix tables (`inside/1`–`inside/23`, `outside/1`–`outside/10`);
the secret prompt of each experiment is in `../secret_prompts.json`.

- `members.json`: the 100 suspect texts (`../mimir_wiki/members.json`) rewritten by GPT-4o-mini with the secret prompt.
  The released adapter of the experiment was LoRA-fine-tuned on these rewrites.
- `nonmembers.json`: the 100 reference texts rewritten with the same prompt (not used for training).

Exceptions:

- `inside/4` and `inside/16`: the rewrites the adapters were trained on were not kept; these files were generated
  afterwards with the same secret prompt and the same suspect texts.
