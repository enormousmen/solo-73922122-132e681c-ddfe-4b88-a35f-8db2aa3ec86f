"""Single-job training manager.

One background trainer thread runs the loop below; FastAPI handlers only
manipulate flags / wait on conditions, so slow training or slow checkpoint
saves never block status queries.

Reproducibility design (bit-exact resume):

* ``torch.set_num_threads(1)`` and deterministic algorithms are enabled, so a
  given sequence of ops always produces the same float32 bits on this build.
* A fresh run seeds the global torch RNG once (``torch.manual_seed(seed)``).
  Model init and Dropout then draw from that single stream.
* The per-epoch sample order is a pure function of ``(seed, epoch)`` via a
  dedicated ``random.Random`` instance, so the shuffle of any epoch can be
  re-derived after a restart without storing it.
* Every checkpoint stores: model + Adam state, the torch RNG state, the
  position (``epoch``, ``next_batch``), the full loss history, the resolved
  config, a ``completed`` flag **and the identity of the exact dataset the run
  was trained on** (feature names + float32/int64 values in file order).
* Resume (after pause, or after a process kill + restart) restores all of the
  above and simply continues the same deterministic sequence, so every
  subsequent batch loss and the final parameters are bit-identical to an
  uninterrupted run.

Dataset binding (why losses always belong to the recorded samples):

* A fresh run snapshots the CSV into memory once and records its fingerprint.
  Edits to the file during the run therefore cannot change what is trained on
  and cannot change the identity stored in subsequent checkpoints.
* On restart, the checkpoint's fingerprint is compared with the CSV *now* at
  the configured path.  Only an exact match resumes.  A changed value, a
  reordered row, a missing file, an unparseable file, or a legacy checkpoint
  without an identity all block resume with a queryable reason; the old
  checkpoint is left untouched on disk.
* ``POST /train/start`` with ``force=true`` moves the old checkpoint (and a
  manifest describing why) into ``<runs_dir>/archive/`` before starting a new
  run on the *current* CSV.
"""
from __future__ import annotations

import io
import json
import os
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import (
    CheckpointError,
    checkpoint_path,
    cleanup_stale_tmp_files,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainConfig
from .dataset import (
    FINGERPRINT_ALGORITHM,
    Dataset,
    DatasetLoadError,
    load_classification_dataset,
)
from .model import build_model


class JobState(str, Enum):
    IDLE = "idle"          # no run exists
    RUNNING = "running"    # trainer thread is working through batches
    PAUSING = "pausing"    # pause requested; finishing batch + checkpoint
    PAUSED = "paused"      # trainer stopped; resumable (see checkpoint_dirty)
    FINISHED = "finished"  # all epochs done and final checkpoint persisted
    FAILED = "failed"      # unrecoverable error (see last_error)


class ConflictError(Exception):
    """409 - request incompatible with current state."""


class NotFoundError(Exception):
    """404 - no run / checkpoint exists."""


class PauseTimeoutError(Exception):
    """504 - trainer did not stop within pause_timeout_seconds."""


class CheckpointSaveError(Exception):
    """500 - pause could not persist a complete checkpoint."""


def epoch_permutation(seed: int, epoch: int, n: int) -> list[int]:
    """Deterministic per-epoch shuffle, re-derivable after a restart."""
    rng = random.Random(f"{seed}:{epoch}")
    perm = list(range(n))
    rng.shuffle(perm)
    return perm


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    """Make an id from an (untrusted) checkpoint safe as a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip("._")
    return cleaned[:64] or "unknown"


class JobManager:
    def __init__(self, runs_dir: str, default_config: TrainConfig):
        # Bit-exact CPU reproducibility: fixed thread count + deterministic ops.
        torch.set_num_threads(1)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:  # pragma: no cover - depends on torch build
            pass

        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.runs_dir / "archive"
        self.default_config = default_config
        # Service-level dataset path (kept even if the file is currently
        # missing, so the reason is reportable instead of crashing at boot).
        self.dataset_path = default_config.dataset_path

        self._cond = threading.Condition()      # guards all mutable fields below
        self._save_lock = threading.Lock()      # serializes checkpoint writes
        self._pause_requested = False
        self._thread: threading.Thread | None = None
        self._resume_rng_state: torch.Tensor | None = None
        self._archived: list[dict] = self._load_archived_manifests()
        self._probe_cache: tuple | None = None
        # Why the checkpoint currently on disk could not be examined (if so).
        self._load_issue: dict | None = None

        self._reset_runtime()
        # A missing/invalid CSV does not prevent the service from starting: the
        # checkpoint stays intact and the reason is visible in GET /train/status.
        self._startup_dataset: Dataset | None = None
        self._startup_dataset_error: str | None = None
        self._recover_from_disk()

    # ------------------------------------------------------------------ setup

    def _reset_runtime(self) -> None:
        self.state = JobState.IDLE
        self.cfg: TrainConfig | None = None
        self.run_id: str | None = None
        self.model = None
        self.optimizer = None
        self.dataset: Dataset | None = None
        # Identity of the data this run is bound to (checkpoint-serialisable).
        self.dataset_info: dict | None = None
        self.epoch = 0            # epoch of the NEXT batch to run
        self.next_batch = 0       # index (within epoch) of the NEXT batch to run
        self.losses: list[dict] = []
        self.completed = False
        self.recovered = False
        self.saving = False
        self.checkpoint_dirty = False   # in-memory progress newer than disk
        self.last_error: str | None = None
        self.last_checkpoint_at: str | None = None
        self.started_at: str | None = None
        self._saved_point: tuple[int, int] | None = None
        self._resume_rng_state = None
        self._data_conflict: dict | None = None
        self._disk_issue: dict | None = None
        self._load_issue: dict | None = None
        self._clear_dataset_fields()

    def _clear_dataset_fields(self) -> None:
        self.num_samples = 0
        self.input_dim = 0
        self.num_classes = 0
        self.feature_cols: list[str] = []

    def _bind_dataset(self, ds: Dataset) -> None:
        """Bind this run to ``ds`` (samples + identity, frozen in memory)."""
        self.dataset = ds
        self.dataset_info = ds.info()
        self.X = ds.X
        self.y = ds.y
        self.feature_cols = list(ds.feature_cols)
        self.num_samples = ds.num_samples
        self.input_dim = ds.num_features
        self.num_classes = ds.num_classes

    def _build_training_objects(self) -> None:
        self.model = build_model(self.input_dim, self.cfg.hidden_dim,
                                 self.cfg.dropout_p, self.num_classes)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    # ------------------------------------------------------- dataset identity

    def _probe_current_dataset(self) -> dict:
        """Load + fingerprint the configured CSV as it is *now* on disk.

        The result is cached against ``(mtime_ns, size)`` so repeated status
        queries do not re-parse an unchanged file, while any edit (including
        one during a run) is still noticed on the next query.
        """
        try:
            st = Path(self.dataset_path).stat()
            cache_key = (str(Path(self.dataset_path).resolve()),
                         st.st_ino, st.st_mtime_ns, st.st_size)
        except OSError:
            cache_key = (str(Path(self.dataset_path).resolve()), None, None, None)
        cached = getattr(self, "_probe_cache", None)
        if cached is not None and cached[0] == cache_key:
            return dict(cached[1])
        if cache_key[1] is None and not Path(self.dataset_path).exists():
            result = {
                "state": "missing",
                "path": self.dataset_path,
                "fingerprint": None,
                "detail": f"dataset file not found: {self.dataset_path}",
            }
        else:
            try:
                ds = load_classification_dataset(self.dataset_path)
                result = {
                    "state": "available",
                    "path": self.dataset_path,
                    "fingerprint": ds.fingerprint,
                    "file_sha256": ds.file_sha256,
                    "file_size": ds.file_size,
                    "num_samples": ds.num_samples,
                    "num_features": ds.num_features,
                    "num_classes": ds.num_classes,
                    "feature_columns": list(ds.feature_cols),
                }
            except DatasetLoadError as exc:
                exists = Path(self.dataset_path).exists()
                result = {
                    "state": "invalid" if exists else "missing",
                    "path": self.dataset_path,
                    "fingerprint": None,
                    "detail": str(exc),
                }
        self._probe_cache = (cache_key, result)
        return dict(result)

    def _dataset_status(self) -> dict:
        """Dataset identity report (caller must hold ``_cond``)."""
        probe = self._probe_current_dataset()
        bound = self.dataset_info
        matches = None
        fingerprint_verified = (bound is not None
                                and isinstance(bound.get("fingerprint"), str))
        if bound is not None and probe.get("fingerprint"):
            matches = bound.get("fingerprint") == probe["fingerprint"]
        if bound is None:
            if self.run_id is not None:
                state = "unverified"
                detail = ("the checkpoint for this run has no recorded dataset "
                          "identity (legacy format); its samples cannot be "
                          "verified")
            else:
                state = "unbound"
                detail = "no dataset identity is bound to a run"
        elif not fingerprint_verified:
            state = "unverified"
            detail = ("the bound run has no recorded dataset identity "
                      "(legacy checkpoint); it cannot be verified")
        elif probe["state"] != "available":
            state = probe["state"]           # "missing" or "invalid"
            detail = probe.get("detail")
        elif matches:
            state = "current"
            detail = None
        else:
            state = "changed"
            detail = ("the CSV on disk no longer matches the samples bound "
                      "to this run (contents and/or row order differ)")
        return {
            "path": self.dataset_path,
            "state": state,
            "detail": detail,
            "identity_rule": FINGERPRINT_ALGORITHM,
            "loaded_in_memory": self.dataset is not None,
            "num_samples": self.num_samples,
            "num_features": self.input_dim,
            "num_classes": self.num_classes,
            "bound": bound,
            "current": probe,
            "matches_bound": matches,
            "startup_error": self._startup_dataset_error,
        }

    def dataset_status(self) -> dict:
        with self._cond:
            return self._dataset_status()

    @staticmethod
    def _conflict(kind: str, reason: str, detail: dict,
                  hint: str | None = None) -> dict:
        return {
            "type": kind,
            "reason": reason,
            "detail": detail,
            "detected_at": _now_iso(),
            "hint": hint or (
                "Restore the original CSV at the configured dataset_path and "
                "restart the service to resume this run, or POST /train/start "
                "with {\"force\": true} to archive the old checkpoint and start "
                "a new run on the current CSV."),
            "old_checkpoint_kept": True,
        }

    def _conflict_message(self) -> str:
        c = self._data_conflict or {}
        return (f"cannot resume: dataset identity is not verified "
                f"({c.get('type')}): {c.get('reason')} {c.get('hint', '')}").strip()

    # ------------------------------------------------------------- recovery

    def _restore_run_metadata(self, ckpt: dict) -> None:
        self.cfg = TrainConfig.from_dict(ckpt["config"])
        self.run_id = str(ckpt["run_id"])
        self.epoch = int(ckpt["epoch"])
        self.next_batch = int(ckpt["next_batch"])
        self.losses = list(ckpt["losses"])
        self.completed = bool(ckpt["completed"])
        self.started_at = ckpt.get("started_at")
        self._saved_point = (self.epoch, self.next_batch)
        info = ckpt.get("dataset")
        if isinstance(info, dict) and isinstance(info.get("fingerprint"), str):
            self.dataset_info = dict(info)
            # Report progress/counts from the identity recorded with the run,
            # not from a possibly-different CSV currently on disk.
            self.num_samples = int(info.get("num_samples", 0) or 0)
            self.input_dim = int(info.get("num_features", 0) or 0)
            self.num_classes = int(info.get("num_classes", 0) or 0)
            self.feature_cols = list(info.get("feature_columns") or [])

    def _recover_from_disk(self) -> None:
        removed = cleanup_stale_tmp_files(self.runs_dir)
        if removed:
            print(f"[trainer] removed stale temp checkpoint files: {removed}")
        if self._thread is not None and self._thread.is_alive():
            # Re-verification while a run is in progress would clobber live
            # state; the trainer's own snapshot is already bound.
            return
        # Re-verification replaces any previous verdict.
        self._data_conflict = None
        self._disk_issue = None
        self._load_issue = None
        # Fresh snapshot of the CSV on disk.  Reading the file exactly once
        # means the samples and their identity always describe the same bytes.
        self._startup_dataset = None
        self._startup_dataset_error = None
        self._probe_cache = None
        try:
            self._startup_dataset = load_classification_dataset(self.dataset_path)
        except DatasetLoadError as exc:
            self._startup_dataset_error = str(exc)
            print(f"[trainer] dataset unavailable: {exc}")
        if not checkpoint_path(self.runs_dir).exists():
            return
        try:
            ckpt = load_checkpoint(self.runs_dir)
        except Exception as exc:  # noqa: BLE001 - torn/unreadable file
            self._disk_issue = {
                "type": "checkpoint_unreadable",
                "reason": f"checkpoint.pt exists but cannot be read: {exc!r}",
                "detail": {"path": str(checkpoint_path(self.runs_dir))},
                "detected_at": _now_iso(),
            }
            self._load_issue = dict(self._disk_issue)
            print(f"[trainer] {self._disk_issue['reason']}")
            return
        try:
            self._restore_run_metadata(ckpt)
        except Exception as exc:  # noqa: BLE001 - malformed payload
            self._reset_runtime()
            self._disk_issue = {
                "type": "checkpoint_invalid",
                "reason": f"checkpoint.pt has an unusable payload: {exc!r}",
                "detail": {"path": str(checkpoint_path(self.runs_dir))},
                "detected_at": _now_iso(),
            }
            self._load_issue = dict(self._disk_issue)
            print(f"[trainer] {self._disk_issue['reason']}")
            return

        try:
            mtime = checkpoint_path(self.runs_dir).stat().st_mtime
            self.last_checkpoint_at = datetime.fromtimestamp(
                mtime, timezone.utc).isoformat()
        except OSError:
            pass

        info = ckpt.get("dataset")
        info_valid = (isinstance(info, dict)
                      and isinstance(info.get("fingerprint"), str))

        if info_valid:
            # Report progress/counts from the identity recorded with the run,
            # not from a possibly-different CSV currently on disk.
            self.num_samples = int(info.get("num_samples", 0) or 0)
            self.input_dim = int(info.get("num_features", 0) or 0)
            self.num_classes = int(info.get("num_classes", 0) or 0)
            self.feature_cols = list(info.get("feature_columns") or [])
        elif self._startup_dataset is not None:
            # Legacy payload: the sample count is unknown.  Show the current
            # file's shape for context only (it is NOT treated as verified).
            ds = self._startup_dataset
            self.num_samples = ds.num_samples
            self.input_dim = ds.num_features
            self.num_classes = ds.num_classes
            self.feature_cols = list(ds.feature_cols)

        if not info_valid:
            self._data_conflict = self._conflict(
                "legacy_checkpoint",
                "checkpoint has no dataset identity (legacy format); the "
                "samples it was trained on cannot be verified, so resuming it "
                "would silently mix training data.",
                {"run_id": self.run_id, "checkpoint_dataset": None,
                 "checkpoint_path": str(checkpoint_path(self.runs_dir))},
                hint=("Legacy checkpoints cannot be verified. POST /train/start "
                      "with {\"force\": true} archives this checkpoint (kept in "
                      "the runs archive/ directory) and starts a new run on the "
                      "current CSV."))
        elif self._startup_dataset is None:
            self._data_conflict = self._conflict(
                "dataset_unavailable",
                f"the CSV the checkpoint was trained on is unavailable at "
                f"{self.dataset_path!r}: {self._startup_dataset_error}",
                {"run_id": self.run_id,
                 "checkpoint_dataset": self.dataset_info,
                 "startup_error": self._startup_dataset_error})
        elif self._startup_dataset.fingerprint != info["fingerprint"]:
            self._data_conflict = self._conflict(
                "dataset_changed",
                "the CSV at the configured path does not match the samples "
                "this checkpoint was trained on (contents or row order changed).",
                {"run_id": self.run_id,
                 "checkpoint_dataset": self.dataset_info,
                 "current_dataset": self._startup_dataset.info()})
        else:
            # Verified: the checkpoint's identity equals the loaded samples.
            self._bind_dataset(self._startup_dataset)
            self._build_training_objects()
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            torch.set_rng_state(ckpt["rng_state"])
            self._resume_rng_state = ckpt["rng_state"]
            self.state = JobState.FINISHED if self.completed else JobState.PAUSED
            self.recovered = True
            print(f"[trainer] recovered checkpoint run={self.run_id} "
                  f"epoch={self.epoch} next_batch={self.next_batch} "
                  f"completed={self.completed} "
                  f"dataset={self.dataset_info['fingerprint'][:23]}...")
            return

        # Verified identity failed: keep run metadata for queries, but never
        # build a model, load an RNG state or allow training to continue.
        self.state = JobState.FINISHED if self.completed else JobState.PAUSED
        self.recovered = True
        self._resume_rng_state = None
        print(f"[trainer] checkpoint run={self.run_id} is NOT resumable: "
              f"{self._data_conflict['type']} - "
              f"{self._data_conflict['reason']}")

    # --------------------------------------------------------------- commands

    def start(self, overrides: dict | None = None, force: bool = False) -> dict:
        with self._cond:
            if self.state in (JobState.RUNNING, JobState.PAUSING):
                raise ConflictError("training is already running")
            cfg = self.default_config.merged(overrides or {})
            if cfg.dataset_path != self.default_config.dataset_path:
                raise ValueError("dataset_path is service-level and cannot be overridden")

            # Load the current CSV *before* touching the old checkpoint: if the
            # dataset cannot be read, nothing is archived and nothing starts.
            try:
                ds = load_classification_dataset(self.dataset_path)
            except DatasetLoadError as exc:
                raise ValueError(
                    f"cannot start training: {exc}") from exc

            if not force:
                if self._data_conflict is not None:
                    raise ConflictError(
                        "a previous run exists but cannot be resumed: "
                        f"{self._data_conflict['reason']} "
                        f"{self._data_conflict['hint']}")
                if self._disk_issue is not None:
                    raise ConflictError(
                        f"{self._disk_issue['reason']}; use POST /train/start "
                        "with force=true to archive it and start a new run")
                if self.run_id is not None:
                    raise ConflictError(
                        "a run already exists; use POST /train/resume to continue it "
                        "or POST /train/start with force=true to discard it")
            else:
                # Force never silently destroys evidence: archive the old
                # checkpoint (and a manifest with the reason) first.
                if checkpoint_path(self.runs_dir).exists():
                    self._archive_checkpoint()

            self._reset_runtime()
            self.cfg = cfg
            self._bind_dataset(ds)               # freeze samples + identity
            torch.manual_seed(cfg.seed)          # seed model init + dropout stream
            self._build_training_objects()
            self.run_id = uuid.uuid4().hex[:12]
            self.started_at = _now_iso()
            self.state = JobState.RUNNING
            self._pause_requested = False
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            return self.status()

    def resume(self) -> dict:
        with self._cond:
            if self.state in (JobState.RUNNING, JobState.PAUSING):
                raise ConflictError("training is already running")
            if self.run_id is None or self.cfg is None:
                if self._disk_issue is not None:
                    raise ConflictError(
                        f"{self._disk_issue['reason']}; use POST /train/start "
                        "with force=true to archive it and start a new run")
                raise NotFoundError("no run or checkpoint to resume from")
            if self._data_conflict is not None:
                # The CSV may have been restored since startup: re-verify the
                # checkpoint against the data on disk once before refusing.
                # (For a legacy/unverifiable checkpoint this cannot succeed.)
                self._recover_from_disk()
            if self._data_conflict is not None:
                raise ConflictError(self._conflict_message())
            if self.state == JobState.FINISHED:
                raise ConflictError(
                    "run already finished; start a new one with force=true")
            if self._disk_issue is not None:
                raise ConflictError(
                    f"{self._disk_issue['reason']}; use POST /train/start "
                    "with force=true to archive it and start a new run")
            if self.dataset is None:
                # Defensive: never train on data whose identity is unproven.
                raise ConflictError(
                    "no verified dataset is loaded for this run; refusing to "
                    "resume")
            # PAUSED or FAILED: continue from the in-memory state, which is the
            # checkpoint state (or newer, if the last save failed).
            if self._resume_rng_state is not None:
                torch.set_rng_state(self._resume_rng_state)
            self.state = JobState.RUNNING
            self._pause_requested = False
            self.last_error = None
            self._thread = threading.Thread(target=self._train_loop,
                                            name="trainer", daemon=True)
            self._thread.start()
            return self.status()

    def pause(self) -> dict:
        """Stop training and return only after a full checkpoint is persisted."""
        do_inline_retry = False
        with self._cond:
            if self.state == JobState.RUNNING:
                self._pause_requested = True
                self.state = JobState.PAUSING
            elif self.state == JobState.PAUSED and self.checkpoint_dirty:
                do_inline_retry = True      # retry the failed pause-save
            elif self.state == JobState.PAUSED:
                return self.status()        # already paused and persisted
            elif self.state == JobState.PAUSING:
                pass                        # concurrent pause: just wait
            else:
                raise ConflictError(f"cannot pause while state={self.state.value}")

            if not do_inline_retry:
                timeout = self.cfg.pause_timeout_seconds if self.cfg else 30.0
                deadline = time.monotonic() + timeout
                while self.state == JobState.PAUSING:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PauseTimeoutError(
                            "trainer did not stop within "
                            f"{timeout}s; pause is still in progress, "
                            "poll GET /train/status")
                    self._cond.wait(remaining)
                if self.state == JobState.FINISHED:
                    # The last batch completed before the pause landed; the run
                    # is stopped and the final checkpoint is persisted.
                    return self.status()

        # State is PAUSED here.  If the trainer's pause-save failed, retry it
        # once inline (outside the lock, so queries stay responsive).
        if self.checkpoint_dirty:
            try:
                self._save_checkpoint()
            except CheckpointError as exc:
                raise CheckpointSaveError(
                    f"training is stopped but the checkpoint could not be "
                    f"persisted; the previous checkpoint is intact on disk and "
                    f"the pause can be retried. cause: {exc}") from exc
        return self.status()

    # ------------------------------------------------------------- train loop

    def _train_loop(self) -> None:
        try:
            cfg = self.cfg
            n = self.num_samples
            bs = cfg.batch_size
            batches_per_epoch = (n + bs - 1) // bs

            while self.epoch < cfg.epochs:
                epoch = self.epoch
                perm = epoch_permutation(cfg.seed, epoch, n)
                for b in range(self.next_batch, batches_per_epoch):
                    idx = perm[b * bs:(b + 1) * bs]
                    xb = self.X[idx]
                    yb = self.y[idx]
                    self.model.train()
                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(xb)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    self.optimizer.step()
                    loss_value = loss.item()
                    if cfg.batch_sleep_seconds > 0:
                        time.sleep(cfg.batch_sleep_seconds)

                    with self._cond:
                        self.next_batch = b + 1
                        self.losses.append({
                            "step": len(self.losses) + 1,
                            "epoch": epoch,
                            "batch": b,
                            "loss": loss_value,
                        })
                        pause_now = self._pause_requested

                    if (b + 1) % cfg.checkpoint_every == 0 or b == batches_per_epoch - 1:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # recorded in status; retried at next boundary

                    if pause_now:
                        try:
                            self._save_checkpoint()
                        except CheckpointError:
                            pass  # pause() reports the failure to the caller
                        with self._cond:
                            self._resume_rng_state = torch.get_rng_state()
                            self._pause_requested = False
                            self.state = JobState.PAUSED
                            self._thread = None      # allow re-verification
                            self._cond.notify_all()
                        return

                with self._cond:
                    self.epoch = epoch + 1
                    self.next_batch = 0

            with self._cond:
                self.completed = True
            try:
                self._save_checkpoint(force=True)
            except CheckpointError:
                with self._cond:
                    self.state = JobState.FAILED
                    self._thread = None
                    self._cond.notify_all()
                return
            with self._cond:
                self.state = JobState.FINISHED
                self._thread = None
                self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - never lose the error
            with self._cond:
                self.last_error = f"training failed: {exc!r}"
                self.state = JobState.FAILED
                self._thread = None
                self._cond.notify_all()

    # -------------------------------------------------------------- checkpoint

    def _save_checkpoint(self, force: bool = False) -> None:
        point = (self.epoch, self.next_batch)
        # The identity comes from the run's bound snapshot, never from a fresh
        # read of the file: an edit during the run cannot change what the
        # checkpoint claims it was trained on.
        payload = {
            "schema_version": 2,
            "run_id": self.run_id,
            "config": self.cfg.to_dict(),
            "epoch": self.epoch,
            "next_batch": self.next_batch,
            "completed": self.completed,
            "losses": self.losses,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "started_at": self.started_at,
            "saved_at": _now_iso(),
            "dataset": self.dataset_info,
            "dataset_fingerprint": (self.dataset_info or {}).get("fingerprint"),
        }
        with self._save_lock:
            if not force and not self.checkpoint_dirty and self._saved_point == point:
                return  # this exact point is already persisted
            with self._cond:
                self.saving = True
            try:
                save_checkpoint(self.runs_dir, payload,
                                save_delay_seconds=self.cfg.checkpoint_sleep_seconds)
            except CheckpointError as exc:
                with self._cond:
                    self.saving = False
                    self.checkpoint_dirty = True
                    self.last_error = (f"checkpoint save failed at "
                                       f"epoch={point[0]} batch={point[1]}: {exc}")
                raise
            with self._cond:
                self.saving = False
                self.checkpoint_dirty = False
                self.last_error = None
                self._saved_point = point
                self.last_checkpoint_at = _now_iso()

    # ---------------------------------------------------------------- archive

    def _load_archived_manifests(self) -> list[dict]:
        if not self.archive_dir.is_dir():
            return []
        manifests = []
        for path in self.archive_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(data, dict):
                data.setdefault("manifest", path.name)
                manifests.append(data)
        manifests.sort(key=lambda m: str(m.get("archived_at", "")), reverse=True)
        return manifests[:50]

    def _archive_checkpoint(self) -> dict | None:
        """Move checkpoint.pt (plus a reason manifest) into ``archive/``.

        Caller must hold ``_cond``.  Returns the manifest (or None if there was
        no checkpoint file).  This is the only place a run is discarded, and it
        never deletes data.
        """
        src = checkpoint_path(self.runs_dir)
        if not src.exists():
            return None
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        run_id = _safe_name(self.run_id or "unknown")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        dest = self.archive_dir / f"checkpoint.{run_id}.{stamp}.pt"
        os.replace(src, dest)
        conflict = self._data_conflict
        issue = self._load_issue
        if conflict is not None:
            reason_type = conflict["type"]
            reason = conflict["reason"]
        elif issue is not None:
            reason_type = issue["type"]
            reason = issue["reason"]
        elif self.run_id is not None:
            reason_type = "force_start"
            reason = "previous run discarded by explicit POST /train/start force=true"
        else:
            reason_type = "unknown"
            reason = "checkpoint archived by an explicit force start"
        manifest = {
            "archived_at": _now_iso(),
            "reason_type": reason_type,
            "reason": reason,
            "run_id": self.run_id,
            "payload_readable": issue is None,
            "checkpoint_file": dest.name,
            "checkpoint_path": str(dest),
            "run_dataset": self.dataset_info,
            "run_progress": {"epoch": self.epoch, "next_batch": self.next_batch,
                             "completed": self.completed},
            "replaced_by_dataset": {
                "path": self.dataset_path,
                "fingerprint": self._probe_current_dataset().get("fingerprint"),
            },
        }
        manifest_path = dest.with_suffix(".json")
        tmp = manifest_path.with_name(manifest_path.name + f".tmp.{uuid.uuid4().hex}")
        try:
            tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, manifest_path)
        except OSError as exc:  # pragma: no cover - best effort metadata
            print(f"[trainer] could not write archive manifest {manifest_path}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        self._archived.insert(0, {**manifest, "manifest": manifest_path.name})
        self._archived = self._archived[:50]
        print(f"[trainer] archived checkpoint run={run_id} -> {dest} "
              f"({reason_type})")
        return manifest

    def archived_checkpoints(self) -> list[dict]:
        with self._cond:
            return [dict(m) for m in self._archived]

    # ----------------------------------------------------------------- queries

    def status(self) -> dict:
        with self._cond:
            if self.cfg is not None:
                bpe = (self.num_samples + self.cfg.batch_size - 1) // self.cfg.batch_size
                total = self.cfg.epochs * bpe
                done = self.epoch * bpe + self.next_batch
            else:
                bpe = total = done = 0
            ready = (self.state == JobState.PAUSED
                     and self._data_conflict is None
                     and self.dataset is not None)
            return {
                "state": self.state.value,
                "run_id": self.run_id,
                "recovered_after_restart": self.recovered,
                "ready_to_resume": ready,
                "epoch": self.epoch,
                "next_batch": self.next_batch,
                "batches_per_epoch": bpe,
                "total_batches": total,
                "completed_batches": done,
                "progress": (done / total) if total else 0.0,
                "saving": self.saving,
                "checkpoint_dirty": self.checkpoint_dirty,
                "last_checkpoint_at": self.last_checkpoint_at,
                "last_error": self.last_error,
                "started_at": self.started_at,
                "config": self.cfg.to_dict() if self.cfg else None,
                "dataset": self._dataset_status(),
                "data_conflict": dict(self._data_conflict)
                if self._data_conflict else None,
                "checkpoint_issue": dict(self._disk_issue)
                if self._disk_issue else None,
                "archived_checkpoints": [dict(m) for m in self._archived],
            }

    def get_losses(self, since_step: int | None = None, limit: int | None = None) -> dict:
        with self._cond:
            losses = list(self.losses)
            total = len(losses)
        if since_step is not None:
            losses = [l for l in losses if l["step"] > since_step]
        if limit is not None:
            losses = losses[:limit]
        return {"count": total, "returned": len(losses), "losses": losses}

    def read_parameters(self) -> bytes:
        """Model parameters from the latest persisted checkpoint, as a .pt file."""
        with self._cond:
            issue = self._disk_issue
        if issue is not None:
            raise ConflictError(issue["reason"])
        try:
            ckpt = load_checkpoint(self.runs_dir)
        except Exception as exc:  # noqa: BLE001 - unreadable checkpoint file
            raise ConflictError(f"checkpoint.pt cannot be read: {exc!r}") from exc
        if ckpt is None:
            raise NotFoundError("no checkpoint exists yet; train first")
        buf = io.BytesIO()
        torch.save(ckpt["model_state_dict"], buf)
        return buf.getvalue()

    def checkpoint_file(self) -> Path:
        path = checkpoint_path(self.runs_dir)
        if not path.exists():
            raise NotFoundError("no checkpoint exists yet; train first")
        return path
