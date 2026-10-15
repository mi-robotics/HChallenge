"""Record demos with the D405.

  python replay/record.py [out_dir]     (default: data/demos)
      s = start, e = end (saves the demo), x = discard, q = quit

Each demo goes to <out_dir>/<YYYYmmdd_HHMMSS>/: color.mp4, depth.mkv (lossless uint16), frames.csv, meta.json.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
from d405 import D405  # noqa: E402
from markers import marker_pose  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "data" / "demos"
INSTRUCTION = "put the cup on the plate"
FPS = 30


def ffmpeg_writer(path, size, pix_fmt_in, codec_args):
    w, h = size
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", pix_fmt_in,
           "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", *codec_args, str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


class Recording:
    def __init__(self, cam, out_dir):
        self.dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
        self.dir.mkdir(parents=True)
        self.cam = cam
        self.color = ffmpeg_writer(self.dir / "color.mp4", cam.size, "bgr24",
                                   ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p"])
        self.depth = ffmpeg_writer(self.dir / "depth.mkv", cam.size, "gray16le",
                                   ["-c:v", "ffv1", "-level", "3", "-threads", "4"])
        self.rows = []
        self.t_start = time.time()

    def add(self, color, depth, ts, fn):
        self.color.stdin.write(color.tobytes())
        self.depth.stdin.write(depth.tobytes())
        self.rows.append((len(self.rows), fn, ts))

    def close(self):
        for p in (self.color, self.depth):
            p.stdin.close()
            p.wait()
        fns = np.array([r[1] for r in self.rows])
        dropped = int((np.diff(fns) - 1).clip(0).sum()) if len(fns) > 1 else 0
        with open(self.dir / "frames.csv", "w") as f:
            f.write("index,frame_number,timestamp_ms\n")
            f.writelines(f"{i},{fn},{ts:.3f}\n" for i, fn, ts in self.rows)
        meta = {
            "instruction": INSTRUCTION,
            "fps": FPS,
            "frames": len(self.rows),
            "dropped_frames": dropped,
            "duration_s": round((self.rows[-1][2] - self.rows[0][2]) / 1000, 3) if self.rows else 0,
            "image_size": list(self.cam.size),
            "camera_matrix": self.cam.K.tolist(),
            "dist_coeffs": self.cam.dist.tolist(),
            "depth_scale": self.cam.depth_scale,
        }
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"saved {self.dir.name}: {meta['frames']} frames, {meta['duration_s']} s, "
              f"{dropped} dropped")

    def discard(self):
        for p in (self.color, self.depth):
            p.stdin.close()
            p.wait()
        shutil.rmtree(self.dir)
        print("discarded", self.dir.name)


def main(out_dir):
    count = lambda: sum(1 for p in out_dir.iterdir() if p.is_dir()) if out_dir.exists() else 0
    n_saved = count()
    cam = D405()
    rec, marker_ok, i = None, False, 0
    print(f"saving to {out_dir} ({n_saved} already there). s = start, e = end, x = discard, q = quit")
    try:
        while True:
            color, depth, ts, fn = cam.read_raw()
            if rec:
                rec.add(color, depth, ts, fn)
            if i % 10 == 0:  # marker check every 1/3 s keeps the loop at 30 fps
                marker_ok = marker_pose(color, cam.K, cam.dist) is not None
            i += 1

            vis = cv2.resize(color, (cam.size[0] // 2, cam.size[1] // 2))
            cv2.putText(vis, "marker OK" if marker_ok else "MARKER NOT VISIBLE", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if marker_ok else (0, 0, 255), 2, cv2.LINE_AA)
            if rec:
                cv2.circle(vis, (vis.shape[1] - 25, 22), 9, (0, 0, 255), -1)
                cv2.putText(vis, f"REC {time.time() - rec.t_start:4.1f} s  (e = end, x = discard)", (vis.shape[1] - 330, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, f"{n_saved} saved in {out_dir.name}   s = start   q = quit", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("record demo", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") and rec is None:
                rec = Recording(cam, out_dir)
                print("recording", rec.dir.name)
            elif key == ord("e") and rec is not None:
                rec.close()
                rec, n_saved = None, count()
            elif key == ord("x") and rec is not None:
                rec.discard()
                rec = None
            elif key == ord("q"):
                break
    finally:
        if rec is not None:
            rec.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEMOS)
