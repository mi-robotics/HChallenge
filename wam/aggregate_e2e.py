"""End-to-end results (WAM plan -> policy) over plan seeds x policy seeds.

  python wam/aggregate_e2e.py [policy run]     (default full_wrist_state_no_img_pos)

Per WAM (v1, v2, v2 + ground truth first flow row), with and without pauses: success % per (plan seed, policy seed),
then mean +- std. Test pools val and test (5 plan seeds x 5 policy seeds); train and failed use plan seed 0.
Writes outputs/wam/e2e_summary[_<policy>].json.
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
run = ROOT / "outputs/s1/ablation" / POLICY
rec_file = run / "eval_ckpt_20000_x5.json"  # 5 attempts per seed, as on the Policy page; else the 1-attempt eval
rec_file = rec_file if rec_file.exists() else run / "eval_ckpt_20000.json"
if rec_file.exists():  # not there with downloaded weights only
    rec = json.loads(rec_file.read_text())["per_episode"]
    out["recorded flow"] = {"test": row(rates(rec, ["val", "test"])), "train": row(rates(rec, ["train"])),
                            "failed": row(rates(rec, ["failed"]))}
for name, gen, d in GENS:
    if not (ROOT / "outputs/wam" / gen / d).exists():  # e.g. only v1 trained
        continue
    for tag in ("", "_pauses"):
        if not (ROOT / "outputs/wam" / gen / d / f"eval_{POLICY}{tag}_train-failed_5seeds.json").exists():
            continue  # scripts/eval_wam.sh runs only the pauses evals
        test = []
        for k in range(5):
            p = ROOT / "outputs/wam" / gen / (d if k == 0 else f"{d}_s{k}") / f"eval_{POLICY}{tag}_val-test_5seeds.json"
            test += rates(json.loads(p.read_text())["per_episode"], ["val", "test"])
        tf = json.loads((ROOT / "outputs/wam" / gen / d / f"eval_{POLICY}{tag}_train-failed_5seeds.json").read_text())["per_episode"]
        out[name + (" + pauses" if tag else "")] = {"test": row(test), "train": row(rates(tf, ["train"])),
                                                   "failed": row(rates(tf, ["failed"]))}
(ROOT / "outputs/wam" / ("e2e_summary.json" if POLICY == DEFAULT else f"e2e_summary_{POLICY}.json")).write_text(json.dumps(out, indent=1))
print(f"{'':28s}{'train':>16s}{'test (val+test)':>20s}{'failed':>16s}")
for k, v in out.items():
    print(f"{k:28s}" + "".join(f"{v[s]['mean']:10.1f} ± {v[s]['std']:4.1f}" + ("  " if s != "failed" else "") for s in ("train", "test", "failed"))
          + f"   runs {v['train']['runs']}/{v['test']['runs']}")
