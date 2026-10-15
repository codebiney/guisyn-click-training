"""Start training in a detached process and persist its PID and log."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="runs/smolvlm2-guisyn")
parser.add_argument("--trainer", choices=["train_small.py", "train_coordinates.py"], default="train_small.py")
args, trainer_args = parser.parse_known_args()
run = root / args.output
run.mkdir(parents=True, exist_ok=True)
pidfile = run / "train.pid"
if pidfile.exists():
    pid = int(pidfile.read_text())
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit(f"Training already active: PID {pid}")
env = os.environ.copy()
env["PYTHONPATH"] = str(root / "vendor")
env["HF_HOME"] = str(root / "data/huggingface")
with (run / "train.log").open("a") as log:
    process = subprocess.Popen([sys.executable, "-u", str(root / "training" / args.trainer),
        "--output", args.output, *trainer_args],
        cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True)
pidfile.write_text(str(process.pid))
print(f"Started PID {process.pid}; log: {run / 'train.log'}")
