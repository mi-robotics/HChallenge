"""Hand and mug flow for every recorded episode, kept and failed (step 4.4).

  python flow/run_flows.py [dataset_dir ...]     (default: data/mug_pnp_dataset data/mug_pnp_failed)
  python flow/run_flows.py --video out.mp4 <episode> ...   tile the episodes' flow_overlay.mp4 with a legend

Per episode with a hand track (hand.npz): SAM 2 mug masks and timing (flow/mug_sam.py) -> CoTracker hand and mug
points (flow/track_points.py) -> flow.npz (flow/to_flow.py) -> flow_overlay.mp4 (flow/visualise.py). A stage whose
output exists is skipped; an episode that raises is recorded with the stage and error and the run goes on.
Writes <dataset>/flow_summary.json.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flow"))
from cotracker.predictor import CoTrackerPredictor  # noqa: E402
from mug_sam import load_models, segment  # noqa: E402
from to_flow import to_flow  # noqa: E402
from track_points import CHECKPOINT, tile, track  # noqa: E402
from visualise import LEGEND, visualise  # noqa: E402


def has_timing(ep):
    return (ep / "mug_mask.npz").exists() and "t_grasp" in np.load(ep / "mug_mask.npz")


def run(ep, models):
    predictor, generator, cotracker = models
    row = {"episode": ep.name, "dataset": ep.parent.name, "stage": None, "error": None}
    stages = [("mug_sam", has_timing, lambda: segment(ep, predictor, generator)),
              ("track_points", lambda e: (e / "cotracker.npz").exists() and has_timing(e)
               and (e / "cotracker.npz").stat().st_mtime > (e / "mug_mask.npz").stat().st_mtime,
               lambda: track(ep, cotracker)),
              ("to_flow", lambda e: (e / "flow.npz").exists() and (e / "flow.npz").stat().st_mtime > (e / "cotracker.npz").stat().st_mtime,
               lambda: to_flow(ep)),
              ("visualise", lambda e: (e / "flow_overlay.mp4").exists() and (e / "flow_overlay.mp4").stat().st_mtime > (e / "flow.npz").stat().st_mtime,
               lambda: visualise(ep))]
    for name, done, fn in stages:
        row["stage"] = name
        if done(ep):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001  (record and move on)
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            return row
    row["stage"] = "done"
    m, f = np.load(ep / "mug_mask.npz"), np.load(ep / "flow.npz")
    row.update(mug_frame=int(m["frame"]), t_grasp=float(m["t_grasp"]), t_release=float(m["t_release"]),
               steps=int(f["t"].shape[0]), hand_visible=float(f["hand"][..., 3].mean()),
               mug_visible=float(f["cup"][..., 3].mean()))
    return row


if __name__ == "__main__":
    if sys.argv[1:2] == ["--video"]:
        out, episodes = Path(sys.argv[2]), [Path(e) for e in sys.argv[3:]]
        out.parent.mkdir(parents=True, exist_ok=True)
        seconds = max(json.loads((e / "meta.json").read_text())["frames"] for e in episodes) / 30
        tile([e / "flow_overlay.mp4" for e in episodes], out, seconds, LEGEND,
             [f"{'kept' if e.parent.name == 'mug_pnp_dataset' else 'failed in sim'}  {e.name}" for e in episodes])
        sys.exit()
    dirs = [Path(d) for d in sys.argv[1:]] or [ROOT / "data" / "mug_pnp_dataset", ROOT / "data" / "mug_pnp_failed"]
    predictor, generator = load_models()
    models = (predictor, generator, CoTrackerPredictor(checkpoint=str(CHECKPOINT), offline=True, window_len=60).cuda())
    for d in dirs:
        episodes = sorted(p for p in d.iterdir() if p.is_dir() and (p / "hand.npz").exists())
        rows, t0 = [], time.time()
        for k, ep in enumerate(episodes):
            rows.append(run(ep, models))
            print(f"[{d.name} {k + 1}/{len(episodes)}] {ep.name}: {rows[-1]['stage']} {rows[-1]['error'] or ''} | "
                  f"{time.time() - t0:.0f} s", flush=True)
        (d / "flow_summary.json").write_text(json.dumps(rows, indent=2))
        print(f"{d.name}: flow for {sum(r['stage'] == 'done' for r in rows)}/{len(rows)} episodes")
