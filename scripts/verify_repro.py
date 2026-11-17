#!/usr/bin/env python3
"""End-to-end verification of bit-exact resumable training.

Spins up real uvicorn server processes and checks, against an uninterrupted
baseline (identical seed + config):

  A   uninterrupted baseline run (reference)
  B   pause + resume                  -> losses & final params bit-identical to A
  C   hard crash + restart + resume   -> bit-identical to A
  D   two concurrent resumes          -> exactly one 200, one 409; result == A
  E1  checkpoint save failure         -> pause reports 500, old checkpoint kept,
                                         retry succeeds, resume reproduces A
  E2  save failure, then hard crash   -> restart recovers the OLD checkpoint,
                                         resume still reproduces A
  F   slow checkpoint saves           -> status queries stay fast while saving
  G   hard crash in the middle of a save -> restart finds a complete (old or
                                         new) checkpoint, resume reproduces
                                         G-baseline
  H   service basics                  -> /, /docs, 404s before first checkpoint
  I   dataset binding                 -> see scenario_dataset_binding()

Scenarios A–H use the shipped dataset.  Scenario I copies the dataset to a
temp path and exercises checkpoint/data binding over real server restarts:
values changed, row order changed, file deleted, CSV edited during a run,
legacy checkpoints without an identity, forced restart on the new CSV, and
resume after restoring the original data.

Exit code 0 iff every scenario passes.  Run from the repo root:

    python3 scripts/verify_repro.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# Shared by scenarios A/B/C/D/E (bit-exact comparison target).
CFG_MAIN = {"epochs": 5, "batch_size": 32, "batch_sleep_seconds": 0.03,
            "checkpoint_every": 2, "checkpoint_sleep_seconds": 0.0}
# Scenario F: slow saves, short run.
CFG_SLOW_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 1.0}
# Scenario G: kill mid-save, short run.
CFG_KILL_SAVE = {"epochs": 1, "batch_size": 32, "batch_sleep_seconds": 0.03,
                 "checkpoint_every": 1, "checkpoint_sleep_seconds": 0.5}

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, note: str = "") -> None:
    RESULTS.append((name, ok, note))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {note}" if note else ""), flush=True)


# --------------------------------------------------------------------- http

def api(port: int, method: str, path: str, body: dict | None = None,
        timeout: float = 120.0) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"detail": raw.decode(errors="replace")}


def api_bytes(port: int, path: str) -> bytes:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=60) as resp:
        return resp.read()


# ------------------------------------------------------------------ servers

class Server:
    def __init__(self, runs_dir: Path, log_dir: Path, tag: str,
                 config_path: Path | None = None):
        self.runs_dir = runs_dir
        self.port = _free_port()
        self.tag = tag
        log = open(log_dir / f"server-{tag}.log", "wb")
        env = dict(os.environ)
        env["RUNS_DIR"] = str(runs_dir)
        env["CONFIG_PATH"] = str(config_path or ROOT / "config" / "default.json")
        env["ENABLE_TEST_CRASH_ENDPOINT"] = "1"
        self.proc = subprocess.Popen(
            [PY, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        self._log_path = log_dir / f"server-{tag}.log"
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server {self.tag} exited early; log:\n"
                    + self._log_path.read_text(errors="replace")[-3000:])
            try:
                code, _ = api(self.port, "GET", "/train/status", timeout=2)
                if code == 200:
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.3)
        raise RuntimeError(f"server {self.tag} did not become ready")

    def kill(self) -> None:
        """Sudden death via the test-only crash endpoint (os._exit).

        Equivalent to SIGKILL for recovery purposes: the process dies
        immediately, wherever it is (including mid-checkpoint-save), with no
        cleanup.  (Real SIGKILL cannot be used in every environment; Docker
        users can equally use `docker kill`, see README.)
        """
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/test/crash", data=b"", timeout=5)
        except Exception:
            pass  # the connection drops when the process dies
        self.proc.wait(timeout=10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.kill()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ helpers

def wait_state(port: int, want: str, timeout: float = 180.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and st["state"] == want:
            return st
        time.sleep(0.2)
    raise RuntimeError(f"state did not become {want!r} within {timeout}s")


def wait_until(port: int, pred, timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code, st = api(port, "GET", "/train/status")
        if code == 200 and pred(st):
            return st
        time.sleep(0.05)
    raise RuntimeError("condition not met within timeout")


def start_run(port: int, cfg: dict) -> dict:
    code, st = api(port, "POST", "/train/start", {"force": True, "config": cfg})
    assert code == 200, f"start failed: {st}"
    return st


def run_to_finish(port: int, cfg: dict) -> tuple[list, bytes]:
    start_run(port, cfg)
    st = wait_state(port, "finished")
    assert st["completed_batches"] == st["total_batches"], st
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    return losses["losses"], params


def compare_to_baseline(name: str, ref_losses: list, ref_params: bytes,
                        got_losses: list, got_params: bytes) -> bool:
    ok = True
    if got_losses == ref_losses:
        print(f"    losses: {len(got_losses)} entries, exact match")
    else:
        ok = False
        n = min(len(got_losses), len(ref_losses))
        diff = next((i for i in range(n) if got_losses[i] != ref_losses[i]), None)
        print(f"    losses: MISMATCH (len {len(got_losses)} vs {len(ref_losses)}, "
              f"first diff at index {diff})")
        if diff is not None:
            print(f"      got: {got_losses[diff]}\n      ref: {ref_losses[diff]}")
    ref_sd = torch.load(io.BytesIO(ref_params), weights_only=True)
    got_sd = torch.load(io.BytesIO(got_params), weights_only=True)
    if ref_sd.keys() != got_sd.keys():
        ok = False
        print(f"    params: key mismatch {sorted(got_sd)} vs {sorted(ref_sd)}")
    elif all(torch.equal(got_sd[k], ref_sd[k]) for k in ref_sd):
        h_ref = hashlib.sha256(ref_params).hexdigest()[:16]
        h_got = hashlib.sha256(got_params).hexdigest()[:16]
        print(f"    params: all tensors bit-identical "
              f"(sha256 {h_got}{' ==' if h_got == h_ref else ' !='} {h_ref})")
    else:
        ok = False
        bad = [k for k in ref_sd if not torch.equal(got_sd[k], ref_sd[k])]
        print(f"    params: MISMATCH in tensors {bad}")
    record(name, ok)
    return ok


# ------------------------------------------------------------------ scenarios

def scenario_basics(port: int) -> None:
    print("\n== H: service basics (idle server) ==")
    code, _ = api(port, "GET", "/")
    record("H1 GET /", code == 200)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs",
                                    timeout=10) as resp:
            docs_ok = resp.status == 200 and b"swagger" in resp.read().lower()
    except Exception:
        docs_ok = False
    record("H2 GET /docs (Swagger UI)", docs_ok)
    code, _ = api(port, "GET", "/train/parameters")
    record("H3 parameters before any checkpoint -> 404", code == 404)
    code, _ = api(port, "POST", "/train/resume")
    record("H4 resume with no run -> 404", code == 404)
    code, _ = api(port, "POST", "/train/pause")
    record("H5 pause while idle -> 409", code == 409)


def scenario_pause_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== B: pause + resume ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 12)
    t0 = time.monotonic()
    code, st = api(port, "POST", "/train/pause")
    dt = time.monotonic() - t0
    ok = (code == 200 and st["state"] == "paused" and not st["checkpoint_dirty"]
          and st["last_checkpoint_at"] is not None)
    record("B1 pause returns only after checkpoint persisted", ok,
           f"pause latency {dt:.2f}s, stopped at batch {st['completed_batches']}")
    code, st = api(port, "POST", "/train/resume")
    record("B2 resume accepted", code == 200 and st["state"] == "running")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("B3 pause+resume == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_concurrent_resume(port: int, ref: tuple[list, bytes]) -> None:
    print("\n== D: concurrent resume starts exactly one trainer ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 8)
    api(port, "POST", "/train/pause")
    codes: list[int] = []

    def fire():
        code, _ = api(port, "POST", "/train/resume")
        codes.append(code)

    threads = [threading.Thread(target=fire) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record("D1 concurrent resumes -> one 200, one 409",
           sorted(codes) == [200, 409], f"got {sorted(codes)}")
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("D2 resumed run == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)


def scenario_save_failure_retry(port: int, runs_dir: Path,
                                ref: tuple[list, bytes]) -> None:
    print("\n== E1: checkpoint save failure keeps old checkpoint, retry works ==")
    start_run(port, CFG_MAIN)
    wait_until(port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")                      # arm test-only fault injection
    time.sleep(0.3)                           # let any in-flight save finish
    ckpt = runs_dir / "checkpoint.pt"
    hash_before = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    code, body = api(port, "POST", "/train/pause")
    record("E1 pause with failing save -> 500", code == 500,
           str(body.get("detail", ""))[:100])
    code, st = api(port, "GET", "/train/status")
    record("E1 status shows dirty checkpoint + error",
           st["state"] == "paused" and st["checkpoint_dirty"]
           and "injected fault" in (st["last_error"] or ""))
    hash_after = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    record("E1 old checkpoint untouched", hash_before == hash_after)
    flag.unlink()                             # fix the "storage"
    code, st = api(port, "POST", "/train/pause")
    record("E1 retried pause -> 200, checkpoint persisted",
           code == 200 and not st["checkpoint_dirty"])
    code, _ = api(port, "POST", "/train/resume")
    record("E1 resume after retry", code == 200)
    wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    params = api_bytes(port, "/train/parameters")
    compare_to_baseline("E1 run after save-failure == uninterrupted",
                        ref[0], ref[1], losses["losses"], params)


def scenario_slow_save_queries(port: int) -> None:
    print("\n== F: slow checkpoint saves do not block queries ==")
    start_run(port, CFG_SLOW_SAVE)
    wait_until(port, lambda s: s["saving"], timeout=30)
    lat = []
    for _ in range(3):
        t0 = time.monotonic()
        code, _ = api(port, "GET", "/train/status")
        lat.append(time.monotonic() - t0)
    code_l, _ = api(port, "GET", "/train/losses")
    worst = max(lat)
    record("F1 queries respond fast while a 1s save is in flight",
           worst < 1.0 and code == 200 and code_l == 200,
           f"worst status latency {worst*1000:.0f}ms")
    code, st = api(port, "POST", "/train/pause", timeout=60)
    record("F2 pause during slow saves", code == 200 and st["state"] == "paused")
    api(port, "POST", "/train/resume")
    st = wait_state(port, "finished", timeout=120)
    record("F3 slow-save run finishes", st["state"] == "finished")


def scenario_kill_restart(runs_dir: Path, log_dir: Path, server: Server,
                          ref: tuple[list, bytes]) -> Server:
    print("\n== C: hard crash + restart recovers from latest checkpoint ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 10
               and s["last_checkpoint_at"] is not None)
    server.kill()
    record("C1 server crashed mid-run", True)
    server2 = Server(runs_dir, log_dir, "C-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("C2 restart recovers paused state from checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and 0 < st["completed_batches"] < st["total_batches"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("C3 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("C4 kill+restart == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_save_failure_then_kill(runs_dir: Path, log_dir: Path, server: Server,
                                    ref: tuple[list, bytes]) -> Server:
    print("\n== E2: save failure, then hard crash -> old checkpoint used ==")
    start_run(server.port, CFG_MAIN)
    wait_until(server.port, lambda s: s["completed_batches"] >= 6
               and s["last_checkpoint_at"] is not None)
    flag = runs_dir / "INJECT_SAVE_FAILURES"
    flag.write_text("1")
    time.sleep(0.3)
    code, _ = api(server.port, "POST", "/train/pause")
    record("E2 pause with failing save -> 500", code == 500)
    server.kill()                              # die while dirty
    flag.unlink()
    server2 = Server(runs_dir, log_dir, "E2-restarted")
    code, st = api(server2.port, "GET", "/train/status")
    record("E2 restart falls back to old complete checkpoint",
           st["state"] == "paused" and st["recovered_after_restart"]
           and not st["checkpoint_dirty"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("E2 resume after restart", code == 200)
    wait_state(server2.port, "finished")
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("E2 save-fail+kill == uninterrupted", ref[0], ref[1],
                        losses["losses"], params)
    return server2


def scenario_kill_mid_save(runs_dir: Path, log_dir: Path, server: Server) -> Server:
    print("\n== G: hard crash in the middle of a checkpoint save ==")
    print("  (baseline: identical config, uninterrupted)")
    ref_losses, ref_params = run_to_finish(server.port, CFG_KILL_SAVE)
    start_run(server.port, CFG_KILL_SAVE)
    # Crash while a save is in flight AND an earlier checkpoint of this run
    # already exists on disk (completed_batches >= 2 means the save for
    # batch 1 finished and the save for batch >= 2 is now running).
    wait_until(server.port,
               lambda s: s["saving"] and s["completed_batches"] >= 2, timeout=30)
    server.kill()                              # crash inside the save window
    record("G1 server crashed during save", True)
    server2 = Server(runs_dir, log_dir, "G-restarted")
    leftovers = [p.name for p in runs_dir.glob(".checkpoint.pt.tmp.*")]
    record("G2 no torn checkpoint / stale temp cleaned", leftovers == [],
           f"leftovers: {leftovers or 'none'}")
    code, st = api(server2.port, "GET", "/train/status")
    record("G3 restart recovers a complete checkpoint (old or new)",
           st["state"] == "paused" and st["recovered_after_restart"],
           f"recovered at batch {st['completed_batches']}/{st['total_batches']}")
    code, _ = api(server2.port, "POST", "/train/resume")
    record("G3b resume after restart", code == 200)
    wait_state(server2.port, "finished", timeout=120)
    _, losses = api(server2.port, "GET", "/train/losses")
    params = api_bytes(server2.port, "/train/parameters")
    compare_to_baseline("G4 kill-mid-save == uninterrupted", ref_losses, ref_params,
                        losses["losses"], params)
    return server2


# ------------------------------------------------- dataset binding scenario

def _write_cfg(work: Path, dataset_path: Path) -> Path:
    """A service config identical to config/default.json but with a temp CSV."""
    cfg = json.loads((ROOT / "config" / "default.json").read_text())
    cfg["dataset_path"] = str(dataset_path)
    path = work / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(_csv.reader(f))
    return rows[0], rows[1:]


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    import csv as _csv
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def _assert_blocked(name: str, port: int, want_type: str,
                    status_code: int = 409) -> dict:
    code, body = api(port, "POST", "/train/resume")
    detail = str(body.get("detail", ""))
    ok = (code == status_code and want_type in detail
          and "force" in detail)
    record(name, ok, f"{code}: {detail[:90]}")
    return body


def _restart(cur: Server, runs_dir: Path, log_dir: Path, tag: str,
             cfg_path: Path) -> Server:
    """Kill a server and start a fresh process on the same runs dir."""
    cur.kill()
    return Server(runs_dir, log_dir, tag, config_path=cfg_path)


def scenario_dataset_binding(work: Path, log_dir: Path) -> dict:
    """Checkpoint <-> dataset identity binding across real server restarts.

    Three private datasets/runs dirs are used so nothing else is touched:

      * base/      uninterrupted baseline on a copy of the CSV
      * edit/      a run whose CSV is edited while it trains (must be unaffected)
      * conflict/  a paused run used to test changed / reordered / missing CSVs,
                   restore-and-resume, force start on new data, legacy payloads
    """
    print("\n== I: checkpoint is bound to the samples it was trained on ==")
    root = work / "binding"
    root.mkdir()
    cfg = {"epochs": 3, "batch_size": 32, "batch_sleep_seconds": 0.01,
           "checkpoint_every": 1}
    header, rows = _read_csv(ROOT / "data" / "classification.csv")
    original_bytes = (ROOT / "data" / "classification.csv").read_bytes()

    def make_area(name: str) -> tuple[Path, Path, Path]:
        area = root / name
        area.mkdir()
        data = area / "dataset.csv"
        data.write_bytes(original_bytes)
        runs = area / "runs"
        return data, runs, _write_cfg(area, data)

    def restart(cur: Server, tag: str, runs_dir: Path) -> Server:
        cur.kill()
        new = Server(runs_dir, log_dir, tag, config_path=cfg_path)
        live["server"] = new
        return new

    live: dict[str, Server] = {}

    # ---------------------------------------------------- uninterrupted base
    d_base, r_base, cfg_base = make_area("base")
    live["server"] = Server(r_base, log_dir, "I0-base", config_path=cfg_base)
    try:
        wait_state(live["server"].port, "idle")
        ref_losses, ref_params = run_to_finish(live["server"].port, cfg)
        record("I0 uninterrupted baseline on a dataset copy",
               len(ref_losses) == 57, f"{len(ref_losses)} batches")
    finally:
        live["server"].kill()

    # ------------------------------------ CSV edited while a run is training
    d_edit, r_edit, cfg_edit = make_area("edit")
    server = Server(r_edit, log_dir, "I1-edit", config_path=cfg_edit)
    live["server"] = server
    cfg_path = cfg_edit
    try:
        st = start_run(server.port, cfg)["dataset"]
        record("I1 fresh run records a dataset identity",
               st["state"] == "current"
               and st["bound"]["fingerprint"].startswith("sha256:")
               and st["current"]["fingerprint"] == st["bound"]["fingerprint"],
               f"{st['current']['num_samples']} samples, "
               f"{st['bound']['fingerprint'][:19]}...")
        bound_fp = st["bound"]["fingerprint"]
        bound_sha = st["bound"]["file_sha256"]
        wait_until(server.port, lambda s: s["completed_batches"] >= 20)
        edited_rows = [list(r) for r in rows]
        edited_rows[0][0] = str(float(edited_rows[0][0]) + 10.0)
        _write_csv(d_edit, header, edited_rows)
        wait_state(server.port, "finished")
        _, losses = api(server.port, "GET", "/train/losses")
        params = api_bytes(server.port, "/train/parameters")
        record("I2a CSV edit during a run does not change loaded data",
               losses["losses"] == ref_losses and _params_equal(params, ref_params),
               "losses & final params identical to the uninterrupted baseline")
        server = restart(server, "I2-edit-restart", r_edit)
        st = api(server.port, "GET", "/train/status")[1]
        record("I2b startup after the edit still reports the original identity",
               st["dataset"]["state"] == "changed"
               and st["dataset"]["matches_bound"] is False
               and st["dataset"]["bound"]["fingerprint"] == bound_fp
               and st["dataset"]["bound"]["file_sha256"] == bound_sha
               and st["dataset"]["current"]["file_sha256"] != bound_sha,
               "the checkpoint kept the samples it was trained on")
    finally:
        live["server"].kill()

    # ------------------------- paused run vs changed / reordered / missing CSV
    d_c, r_c, cfg_c = make_area("conflict")
    cfg_path = cfg_c
    ckpt_file = r_c / "checkpoint.pt"
    server = Server(r_c, log_dir, "I3-paused", config_path=cfg_c)
    live["server"] = server
    try:
        start_run(server.port, cfg)
        wait_until(server.port, lambda s: s["completed_batches"] >= 10
                   and s["last_checkpoint_at"] is not None)
        code, st = api(server.port, "POST", "/train/pause")
        record("I3a paused run has a complete checkpoint",
               code == 200 and st["state"] == "paused"
               and st["dataset"]["state"] == "current")
        ckpt_hash = hashlib.sha256(ckpt_file.read_bytes()).hexdigest()

        # --- values changed on disk, then restart -------------------------
        changed_rows = [list(r) for r in rows]
        changed_rows[0][0] = str(float(changed_rows[0][0]) + 10.0)
        _write_csv(d_c, header, changed_rows)
        server = restart(server, "I3-changed", r_c)
        st = api(server.port, "GET", "/train/status")[1]
        code, ds = api(server.port, "GET", "/dataset/status")
        record("I3b0 GET /dataset/status reports the change",
               code == 200 and ds["state"] == "changed"
               and ds["matches_bound"] is False
               and ds["bound"]["fingerprint"] != ds["current"]["fingerprint"],
               f"{ds['state']}")
        record("I3b restart detects changed dataset",
               bool(st["data_conflict"])
               and st["data_conflict"]["type"] == "dataset_changed"
               and st["ready_to_resume"] is False
               and st["dataset"]["state"] == "changed",
               st["data_conflict"]["type"] if st["data_conflict"] else "no conflict")
        _assert_blocked("I3c resume refused with a queryable reason", server.port,
                        "dataset_changed")
        record("I3d old checkpoint left untouched",
               hashlib.sha256(ckpt_file.read_bytes()).hexdigest() == ckpt_hash)
        code, _ = api(server.port, "POST", "/train/start", {"config": cfg})
        record("I3e non-force start also refused", code == 409)

        # --- same values, only two rows swapped ---------------------------
        swapped = [list(r) for r in rows]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        _write_csv(d_c, header, swapped)
        server = restart(server, "I3-order", r_c)
        st = api(server.port, "GET", "/train/status")[1]
        record("I4 row reorder alone is a different dataset",
               bool(st["data_conflict"])
               and st["data_conflict"]["type"] == "dataset_changed",
               "row order is part of the identity")
        _assert_blocked("I4b resume refused after row reorder", server.port,
                        "dataset_changed")

        # --- original file deleted ----------------------------------------
        d_c.unlink()
        server = restart(server, "I3-missing", r_c)
        st = api(server.port, "GET", "/train/status")[1]
        record("I5a missing original file is reported",
               bool(st["data_conflict"])
               and st["data_conflict"]["type"] == "dataset_unavailable"
               and st["dataset"]["state"] == "missing"
               and st["dataset"]["loaded_in_memory"] is False,
               st["data_conflict"]["type"] if st["data_conflict"] else "no conflict")
        _assert_blocked("I5b resume refused while the file is missing",
                        server.port, "dataset_unavailable")
        code, _ = api(server.port, "POST", "/train/start", {"config": cfg})
        record("I5c new run refused while the CSV is missing", code == 400)
        code, _ = api(server.port, "POST", "/train/start",
                      {"force": True, "config": cfg})
        record("I5d force start without a readable CSV is refused", code == 400)
        record("I5e old checkpoint still kept untouched",
               hashlib.sha256(ckpt_file.read_bytes()).hexdigest() == ckpt_hash)
        code, body = api(server.port, "GET", "/train/archived")
        record("I5f archived list is queryable", code == 200 and body["count"] == 0)

        # --- restore the original bytes, no restart -> resume works --------
        d_c.write_bytes(original_bytes)
        code, ds = api(server.port, "GET", "/dataset/status")
        record("I5g re-verification on resume notices restored data "
               "(no restart needed)",
               code == 200 and ds["state"] == "current" and ds["matches_bound"],
               f"dataset state after restore: {ds['state']}")
        code, st = api(server.port, "POST", "/train/resume")
        record("I6a restoring the original data allows resume without a restart",
               code == 200 and st.get("state") == "running",
               f"state={st.get('state')}")
        wait_state(server.port, "finished")
        _, losses = api(server.port, "GET", "/train/losses")
        params = api_bytes(server.port, "/train/parameters")
        compare_to_baseline("I6b restored-data resume is bit-exact",
                            ref_losses, ref_params, losses["losses"], params)

        # --- a genuinely new CSV + force start -> archive + new run -------
        new_rows = [list(r) for r in rows]
        for r in new_rows:
            r[0] = str(round(float(r[0]) * 0.5, 6))
            r[1] = str(round(float(r[1]) * 0.5, 6))
        _write_csv(d_c, header, new_rows)
        server = restart(server, "I7-forced", r_c)
        code, st = api(server.port, "POST", "/train/start",
                       {"force": True, "config": cfg})
        record("I7a force start accepted on the current CSV", code == 200,
               f"run={st.get('run_id')}")
        record("I7b new run is bound to the new data",
               st["dataset"]["state"] == "current"
               and st["dataset"]["bound"]["fingerprint"] != bound_fp
               and st["dataset"]["bound"]["file_sha256"]
               == st["dataset"]["current"]["file_sha256"],
               st["dataset"]["bound"]["fingerprint"][:19] + "...")
        _, body = api(server.port, "GET", "/train/archived")
        archived = body.get("archived") or []
        record("I7c old checkpoint archived with a reason (not deleted)",
               len(archived) == 1
               and archived[0]["reason_type"] == "dataset_changed"
               and (r_c / "archive" / archived[0]["checkpoint_file"]).exists(),
               archived[0]["reason_type"] if archived else "nothing archived")
        wait_state(server.port, "finished")
        _, forced = api(server.port, "GET", "/train/losses")
        record("I7d forced run finishes on the new data",
               len(forced["losses"]) == len(ref_losses)
               and forced["losses"] != ref_losses,
               f"{len(forced['losses'])} batches, final loss "
               f"{forced['losses'][-1]['loss']:.6f}")
    finally:
        live["server"].kill()

    # ---------------------------------------- legacy payload without identity
    legacy_area = root / "legacy"
    legacy_area.mkdir()
    legacy_runs = legacy_area / "runs"
    legacy_runs.mkdir()
    legacy_data = legacy_area / "dataset.csv"
    legacy_data.write_bytes(original_bytes)
    legacy_cfg = _write_cfg(legacy_area, legacy_data)
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    ckpt.pop("dataset", None)
    ckpt.pop("dataset_fingerprint", None)
    ckpt["schema_version"] = 1
    torch.save(ckpt, legacy_runs / "checkpoint.pt")
    cfg_path = legacy_cfg
    server = Server(legacy_runs, log_dir, "I8-legacy", config_path=legacy_cfg)
    live["server"] = server
    try:
        st = api(server.port, "GET", "/train/status")[1]
        record("I8a legacy checkpoint is not silently resumed",
               bool(st["data_conflict"])
               and st["data_conflict"]["type"] == "legacy_checkpoint"
               and st["ready_to_resume"] is False
               and st["dataset"]["state"] == "unverified"
               and st["state"] == "finished",
               st["data_conflict"]["type"] if st["data_conflict"] else "no conflict")
        _assert_blocked("I8b legacy resume refused with a reason", server.port,
                        "legacy_checkpoint")
        code, _ = api(server.port, "POST", "/train/start",
                      {"force": True, "config": cfg})
        code_arch, body = api(server.port, "GET", "/train/archived")
        record("I8c force start archives the legacy checkpoint",
               code == 200 and code_arch == 200
               and any(a["reason_type"] == "legacy_checkpoint"
                       for a in body["archived"]),
               f"{body.get('count', 0)} archived")
    finally:
        live["server"].kill()

    # ---------------------------------------- unreadable (torn) checkpoint
    torn_area = root / "torn"
    torn_area.mkdir()
    torn_runs = torn_area / "runs"
    torn_runs.mkdir()
    torn_data = torn_area / "dataset.csv"
    torn_data.write_bytes(original_bytes)
    torn_cfg = _write_cfg(torn_area, torn_data)
    (torn_runs / "checkpoint.pt").write_bytes(b"not a torch checkpoint")
    cfg_path = torn_cfg
    server = Server(torn_runs, log_dir, "I9-torn", config_path=torn_cfg)
    live["server"] = server
    try:
        st = api(server.port, "GET", "/train/status")[1]
        record("I9a unreadable checkpoint is reported, not resumed",
               bool(st["checkpoint_issue"])
               and st["checkpoint_issue"]["type"] == "checkpoint_unreadable"
               and st["state"] == "idle",
               st["checkpoint_issue"]["type"] if st["checkpoint_issue"] else "none")
        code, _ = api(server.port, "POST", "/train/resume")
        record("I9b resume with an unreadable checkpoint -> 409", code == 409)
        code, _ = api(server.port, "POST", "/train/start", {"config": cfg})
        record("I9c non-force start refused", code == 409)
        code, st = api(server.port, "POST", "/train/start",
                       {"force": True, "config": cfg})
        record("I9d force start archives it and starts a new run", code == 200,
               f"run={st.get('run_id')}")
        _, body = api(server.port, "GET", "/train/archived")
        record("I9e archived reason is checkpoint_unreadable",
               any(a["reason_type"] == "checkpoint_unreadable"
                   for a in body["archived"]))
    finally:
        live["server"].kill()

    return {"orig_losses": ref_losses, "orig_params": ref_params}


def _params_equal(a: bytes, b: bytes) -> bool:
    sa = torch.load(io.BytesIO(a), weights_only=True)
    sb = torch.load(io.BytesIO(b), weights_only=True)
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def run_to_finish_after_start(port: int) -> dict:
    st = wait_state(port, "finished")
    _, losses = api(port, "GET", "/train/losses")
    st["losses"] = losses["losses"]
    return st


# ---------------------------------------------------------------------- main

def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="verify-repro-"))
    runs_dir = work / "runs"
    log_dir = work / "logs"
    log_dir.mkdir()
    print(f"workdir: {work}")
    server = Server(runs_dir, log_dir, "main")
    try:
        scenario_basics(server.port)

        print("\n== A: uninterrupted baseline ==")
        ref_losses, ref_params = run_to_finish(server.port, CFG_MAIN)
        record("A baseline finished", True,
               f"{len(ref_losses)} batches, final loss {ref_losses[-1]['loss']:.6f}")

        scenario_pause_resume(server.port, (ref_losses, ref_params))
        scenario_concurrent_resume(server.port, (ref_losses, ref_params))
        scenario_save_failure_retry(server.port, runs_dir, (ref_losses, ref_params))
        scenario_slow_save_queries(server.port)

        server = scenario_kill_restart(runs_dir, log_dir, server,
                                       (ref_losses, ref_params))
        server = scenario_save_failure_then_kill(runs_dir, log_dir, server,
                                                 (ref_losses, ref_params))
        server = scenario_kill_mid_save(runs_dir, log_dir, server)
    finally:
        server.stop()

    try:
        scenario_dataset_binding(work, log_dir)
    except Exception as exc:  # noqa: BLE001 - a crash here is a failed check
        record("I dataset binding scenario crashed", False, repr(exc))
        traceback.print_exc()

    print("\n================ SUMMARY ================")
    failed = 0
    for name, ok, note in RESULTS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        failed += 0 if ok else 1
    print(f"  {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    if failed:
        print(f"  server logs under {log_dir}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
