"""Record and replay the LLM calls of an SDR run so that it can be reproduced exactly.

SDR samples from the LLM (temperature 0.5), so two runs differ. With a cache, every call made through
`get_completion` is stored in order; replaying the cache returns the same answers in the same order, and
with fixed seeds the whole run (and its numbers) is reproduced. The algorithm code is not modified: the
cache replaces the `get_completion` reference in each module that calls it.
"""
import gzip
import hashlib
import json
import os
from collections import defaultdict

from . import llm_api, register_id, sdr_loop, templates

_ORIGINAL = llm_api.get_completion
_CALL_SITES = (llm_api, register_id, sdr_loop, templates)


class GPTCacheMiss(RuntimeError):
    pass


def _key(prompt, model, max_tokens):
    return hashlib.sha256(json.dumps([model, max_tokens, prompt], ensure_ascii=False).encode("utf-8")).hexdigest()


class GPTCache:
    """The n-th occurrence of a request is answered with the n-th recorded answer for that request."""

    def __init__(self, path, strict=False):
        self.path, self.strict = path, strict
        self.recorded = defaultdict(list)
        self.seen = defaultdict(int)
        self.hits = self.misses = 0
        if os.path.exists(path):
            opener = gzip.open if path.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line)
                    self.recorded[record["key"]].append(record["response"])

    def __call__(self, prompt, model="gpt-4o-mini", max_tokens=1024):
        key = _key(prompt, model, max_tokens)
        n = self.seen[key]
        self.seen[key] += 1
        if n < len(self.recorded[key]):
            self.hits += 1
            return self.recorded[key][n]
        if self.strict:
            raise GPTCacheMiss(f"request occurrence #{n + 1} is not in {self.path} (strict replay)")
        if self.path.endswith(".gz"):
            raise GPTCacheMiss(f"{self.path} is compressed and read-only; decompress it to record new calls")
        response = _ORIGINAL(prompt, model=model, max_tokens=max_tokens)
        self.misses += 1
        self.recorded[key].append(response)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "n": n, "model": model, "response": response}, ensure_ascii=False) + "\n")
        return response


def enable(path, strict=False):
    """Route every LLM call made by the SDR code through a record/replay cache stored at `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cache = GPTCache(path, strict=strict)
    for module in _CALL_SITES:
        setattr(module, "get_completion", cache)
    return cache
