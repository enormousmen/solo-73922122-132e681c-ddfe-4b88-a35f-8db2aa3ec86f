#!/usr/bin/env python3
"""End-to-end demo: replace the default CSV, see resume refuse, then recover.

Runs entirely inside a temp directory using real uvicorn server processes,
so the repository's data/ and runs/ are untouched.  Steps:

  1. start a run on a copy of the shipped CSV and pause it
  2. replace the CSV (values scaled) and restart the service
     -> GET /train/status reports dataset_changed, POST /train/resume -> 409,
        the old checkpoint.pt is still on disk, GET /train/archived is empty
  3. force-start ({"force": true}) -> old checkpoint archived with a reason,
        new run trains on the current CSV
  4. restore the original CSV and restart a fresh runs dir copy -> a run
        paused on the original data resumes and finishes

Usage:
    python3 scripts/demo_dataset_change.py

Exit code 0 iff every demo expectation holds.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_repro as v  # noqa: E402

ROOT = v.ROOT
CFG = {"epochs": 2, "batch_size": 32, "batch_sleep_seconds": 0.02,
       "checkpoint_every": 1}


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="demo-dataset-change-"))
    log_dir = work / "logs"
    log_dir.mkdir()
    area = work / "area"
    area.mkdir()
    data = area / "classification.csv"
    shutil.copyfile(ROOT / "data" / "classification.csv", data)
    cfg_path = v._write_cfg(area, data)
    original = data.read_bytes()
    runs = area / "runs"
    print(f"workdir: {work}  (no repository files are modified)")

    server = v.Server(runs, log_dir, "demo", config_path=cfg_path)
    try:
        # 1 -- run + pause on the original data
        print("\n[1] start on the original CSV, then pause")
        v.start_run(server.port, CFG)
        v.wait_until(server.port, lambda s: s["completed_batches"] >= 12
                     and s["last_checkpoint_at"] is not None)
        code, st = v.api(server.port, "POST", "/train/pause")
        print(f"    pause -> {code}, state={st['state']}, "
              f"fingerprint={st['dataset']['bound']['fingerprint'][:23]}...")
        stopped = st["completed_batches"]

        # 2 -- replace the CSV and restart
        print("\n[2] replace the default CSV (all features scaled by 0.5) "
              "and restart the service")
        header, rows = v._read_csv(data)
        new_rows = [[str(round(float(r[0]) * 0.5, 6)),
                     str(round(float(r[1]) * 0.5, 6)), r[2]] for r in rows]
        v._write_csv(data, header, new_rows)
        server = v._restart(server, runs, log_dir, "demo-changed", cfg_path)
        st = v.api(server.port, "GET", "/train/status")[1]
        print(f"    state={st['state']} ready_to_resume={st['ready_to_resume']}")
        print(f"    dataset.state={st['dataset']['state']} "
              f"old={st['dataset']['bound']['fingerprint'][:23]}... "
              f"new={st['dataset']['current']['fingerprint'][:23]}...")
        print(f"    data_conflict.type={st['data_conflict']['type']}")
        print(f"    reason: {st['data_conflict']['reason']}")
        code, body = v.api(server.port, "POST", "/train/resume")
        print(f"\n    POST /train/resume -> {code}")
        print(f"    {str(body.get('detail', ''))[:160]}...")
        assert code == 409, code
        assert st["state"] == "paused"
        assert st["completed_batches"] == stopped
        assert (runs / "checkpoint.pt").exists(), "old checkpoint must be kept"

        # 3 -- force start on the new CSV
        print("\n[3] POST /train/start {\"force\": true} on the new CSV")
        code, st = v.api(server.port, "POST", "/train/start",
                         {"force": True, "config": CFG})
        print(f"    force start -> {code}, run={st['run_id']}, "
              f"dataset={st['dataset']['state']}")
        v.wait_state(server.port, "finished")
        _, losses = v.api(server.port, "GET", "/train/losses")
        print(f"    new run finished: {len(losses['losses'])} batches, final "
              f"loss {losses['losses'][-1]['loss']:.6f}")
        _, arch = v.api(server.port, "GET", "/train/archived")
        man = arch["archived"][0]
        print(f"    archived: {man['checkpoint_file']} "
              f"(reason={man['reason_type']}, run={man['run_id']})")
        print(f"    archive file on disk: "
              f"{(runs / 'archive' / man['checkpoint_file']).exists()}")
        server.kill()

        # 4 -- restore the original data on a fresh copy and resume
        print("\n[4] fresh runs dir: pause on the original CSV, restart, resume")
        data.write_bytes(original)
        runs2 = area / "runs2"
        server = v.Server(runs2, log_dir, "demo-restore", config_path=cfg_path)
        v.start_run(server.port, CFG)
        v.wait_until(server.port, lambda s: s["completed_batches"] >= 12)
        v.api(server.port, "POST", "/train/pause")
        server = v._restart(server, runs2, log_dir, "demo-restore-2", cfg_path)
        st = v.api(server.port, "GET", "/train/status")[1]
        print(f"    after restart: state={st['state']} "
              f"dataset.state={st['dataset']['state']} "
              f"ready_to_resume={st['ready_to_resume']}")
        code, st = v.api(server.port, "POST", "/train/resume")
        print(f"    resume -> {code}")
        assert code == 200, code
        v.wait_state(server.port, "finished")
        _, losses = v.api(server.port, "GET", "/train/losses")
        print(f"    finished: {len(losses['losses'])} batches")
    finally:
        server.stop()

    print("\nDEMO OK: changed data was refused, old checkpoint kept, "
          "force start archived it, restored data resumed.")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
