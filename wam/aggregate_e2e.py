"""Step 6.3 results over plan seeds x policy seeds.

  python wam/aggregate_e2e.py [policy run, default full_wrist_state_no_img_pos]

Per generator (v1, v2, v2 + recorded first flow row) and with / without pauses: success % per (plan seed, policy
seed) over the episodes of a split, then mean +- std over those runs. val and test are pooled as "test" (20
episodes; 5 plan seeds x 5 policy seeds = 25 runs). train and failed use the plan seed 0 plans only (5 runs).
Recorded flow = the step 5 eval of the same policy (5 policy seeds). Writes outputs/wam/e2e_summary.json
(e2e_summary_<policy>.json for another policy).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = "full_wrist_state_no_img_pos"
POLICY = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
GENS = [("v1", "v1", "e2e"), ("v2", "v2_state", "e2e"), ("v2 + first row", "v2_state", "e2e_given")]


def rates(per, splits):
    """per[split][seed][episode] = [ok per attempt] -> success % per policy seed, splits pooled."""
    seeds = per[splits[0]].keys()
    flat = lambda v: v if isinstance(v, list) else [v]  # step 5 evals store one bool per episode
    return [100 * np.mean([a for sp in splits for v in per[sp][s].values() for a in flat(v)]) for s in seeds]


def row(runs):
    return {"mean": float(np.mean(runs)), "std": float(np.std(runs)), "runs": len(runs)}


out = {}
rec = json.loads((ROOT / "outputs/s1/ablation" / POLICY / "eval_ckpt_20000.json").read_text())["per_episode"]
out["recorded flow"] = {"test": row(rates(rec, ["val", "test"])), "train": row(rates(rec, ["train"])),
                        "failed": row(rates(rec, ["failed"]))}
for name, run, d in GENS:
    for tag in ("", "_pauses"):
        test = []
        for k in range(5):
            p = ROOT / "outputs/wam" / run / (d if k == 0 else f"{d}_s{k}") / f"eval_{POLICY}{tag}_val-test_5seeds.json"
            test += rates(json.loads(p.read_text())["per_episode"], ["val", "test"])
        tf = json.loads((ROOT / "outputs/wam" / run / d / f"eval_{POLICY}{tag}_train-failed_5seeds.json").read_text())["per_episode"]
        out[name + (" + pauses" if tag else "")] = {"test": row(test), "train": row(rates(tf, ["train"])),
                                                   "failed": row(rates(tf, ["failed"]))}
(ROOT / "outputs/wam" / ("e2e_summary.json" if POLICY == DEFAULT else f"e2e_summary_{POLICY}.json")).write_text(json.dumps(out, indent=1))
print(f"{'':28s}{'train':>16s}{'test (val+test)':>20s}{'failed':>16s}")
for k, v in out.items():
    print(f"{k:28s}" + "".join(f"{v[s]['mean']:10.1f} ± {v[s]['std']:4.1f}" + ("  " if s != "failed" else "") for s in ("train", "test", "failed"))
          + f"   runs {v['train']['runs']}/{v['test']['runs']}")
