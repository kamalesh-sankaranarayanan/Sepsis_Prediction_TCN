"""One-command local demo: python -B -m src.demo [--check-only]."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .check_demo import main as check_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be between 1024 and 65535")
    if check_ready([]) != 0:
        return 1
    if args.check_only:
        return 0
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
    process = subprocess.Popen([sys.executable, "-B", "-m", "streamlit", "run", "app.py",
                                "--server.address", "127.0.0.1", "--server.port", str(args.port),
                                "--server.headless", "true", "--browser.gatherUsageStats", "false"],
                               cwd=Path(__file__).resolve().parents[1])
    try:
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
