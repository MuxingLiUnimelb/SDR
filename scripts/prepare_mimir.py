"""Export a MIMIR split as the three JSON files SDR expects.

"members" are the texts the launderer rewrites and fine-tunes on (MIMIR *non-member* texts, so the
base model has not seen them), "nonmembers" are held-out texts from the same distribution, and
"prefix" holds the non-member texts used as the ReCaLL prefix.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sdr.data import create_dataset  # noqa: E402
from sdr.utils import save_json, set_seed_all  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="mimir_wiki", choices=["mimir_wiki", "mimir_c4", "mimir_hackernews"])
    p.add_argument("--sub_data", default="ngram_13_0.2")
    p.add_argument("--target_num", type=int, default=200, help="members + nonmembers")
    p.add_argument("--prefix_size", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="data/mimir_wiki")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed_all(args.seed)
    members, nonmembers, prefix, _ = create_dataset(args.dataset, args.sub_data, args.target_num, args.prefix_size)
    save_json(members, os.path.join(args.out_dir, "members.json"))
    save_json(nonmembers, os.path.join(args.out_dir, "nonmembers.json"))
    save_json(prefix, os.path.join(args.out_dir, "prefix.json"))
    print(f"Wrote {len(members)} members, {len(nonmembers)} nonmembers, {len(prefix)} prefix texts to {args.out_dir}")


if __name__ == "__main__":
    main()
