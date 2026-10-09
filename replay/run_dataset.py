"""Run the Phase 1 pipeline and the sim replay on every recorded episode, and count the successes.

  python replay/run_dataset.py [data/mug_pnp_dataset]

Per episode, the same stages as the 5 Phase 1 demos (step 1.4-1.9), without their per-step plots:
  track_hand -> to_table -> clean -> retarget -> scene_match -> replay (all three rim-grasp constraints)
A stage whose output file already exists is skipped, so a rerun picks up where it stopped. An episode that
raises in a stage is recorded with that stage and the error, and the run goes on to the next episode.
Each episode runs in its own process: with MediaPipe and MuJoCo's EGL renderer in one long-running process,
scene_match segfaulted on episode 5 (the same episode runs fine on its own).

Writes <dataset>/replay_summary.json (one row per episode: stage reached, error, the replay's numbers).
"""
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "replay"))
from clean import clean  # noqa: E402
from replay import make_env, replay  # noqa: E402
from retarget import retarget  # noqa: E402
from scene_match import match  # noqa: E402
from to_table import convert  # noqa: E402
from track_hand import track  # noqa: E402

STAGES = [("track_hand", "hand.npz", lambda d, env: track(d)),
          ("to_table", "hand_table.npz", lambda d, env: convert(d)),
          ("clean", "traj.npz", lambda d, env: clean(d)),
          ("retarget", "retarget.npz", lambda d, env: retarget(d)),
          ("scene_match", "scene.json", lambda d, env: match(d, env)),
          ("replay", "replay.json", lambda d, env: replay(d, env))]


def run(episode, env):
    row = {"episode": episode.name, "stage": None, "error": None}
    for name, output, fn in STAGES:
        row["stage"] = name
        if (episode / output).exists():
            continue
        try:
            fn(episode, env)
        except Exception as e:  # noqa: BLE001  (record and move on)
            row["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            return row
    row["stage"] = "done"
    row.update(json.loads((episode / "replay.json").read_text()))
    row.update({k: v for k, v in json.loads((episode / "scene.json").read_text()).items()
                if k in ("plate_minus_cup_cm", "max_reach_m", "min_target_height_above_table_m")})
    return row


if __name__ == "__main__":
    if sys.argv[1:2] == ["--episode"]:  # child: one episode, row as JSON on the last line
        env = make_env(img_size=64)
        row = run(Path(sys.argv[2]), env)
        env.close()
        print(json.dumps(row))
        sys.exit(0)
    dataset = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "mug_pnp_dataset").resolve()
    episodes = sorted(p for p in dataset.iterdir() if p.is_dir())
    rows, t0 = [], time.time()
    for k, ep in enumerate(episodes):
        out = subprocess.run([sys.executable, __file__, "--episode", str(ep)], capture_output=True, text=True)
        try:
            rows.append(json.loads(out.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError):
            rows.append({"episode": ep.name, "stage": "crashed", "error": f"exit {out.returncode}: {out.stderr[-300:]}"})
        ok = sum(r.get("success", False) for r in rows)
        print(f"[{k + 1}/{len(episodes)}] {ep.name}: {rows[-1]['stage']} {'' if rows[-1]['error'] is None else rows[-1]['error']}"
              f" | successes so far {ok}/{k + 1} | {time.time() - t0:.0f} s", flush=True)
    (dataset / "replay_summary.json").write_text(json.dumps(rows, indent=2))
    done = [r for r in rows if r["stage"] == "done"]
    print(f"replayed {len(done)}/{len(rows)}; success {sum(r['success'] for r in done)}/{len(rows)}")
