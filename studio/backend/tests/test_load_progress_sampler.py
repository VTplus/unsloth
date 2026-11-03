# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only tests for the sd-server weight-load sampler (no process, no GPU)."""

import threading
import time

from core.inference import load_progress_sampler as sampler_module


def _patch_readers(monkeypatch, read, rss):
    monkeypatch.setattr(sampler_module, "process_read_bytes", lambda pid: read)
    monkeypatch.setattr(sampler_module, "process_rss_bytes", lambda pid: rss)


def _seed_vram(
    sampler,
    pid,
    value,
    *,
    age = 0.0,
):
    """Fresh cache entry, so snapshot() never starts a background query."""
    sampler._vram_pid = pid
    sampler._vram_value = value
    sampler._vram_at = time.monotonic() - age


def test_model_file_bytes_sums_flagged_files_and_skips_missing(tmp_path):
    diffusion = tmp_path / "model.gguf"
    diffusion.write_bytes(b"x" * 100)
    llm = tmp_path / "llm.gguf"
    llm.write_bytes(b"y" * 50)
    missing = tmp_path / "missing.gguf"
    args = [
        "sd-server.exe",
        "--diffusion-model",
        str(diffusion),
        "--llm",
        str(llm),
        "--vae",
        str(missing),
    ]
    assert sampler_module.model_file_bytes(args) == 150


def test_snapshot_follows_read_share_and_keeps_high_water(monkeypatch):
    sampler = sampler_module.LoadProgressSampler()
    _patch_readers(monkeypatch, read = 80, rss = 20)
    sampler._bind_pid(7)  # first contact, as snapshot() would do
    _seed_vram(sampler, 7, 1234)
    snap = sampler.snapshot(7, None, total_bytes = 100)
    assert snap["fraction"] == 0.8
    assert snap["bytes_loaded"] == 80
    assert snap["phase"] == "loading"
    assert snap["vram_mb"] == 1234
    # RSS drops after the VRAM upload: the high-water mark keeps the bar where it was.
    _patch_readers(monkeypatch, read = 80, rss = 5)
    assert sampler.snapshot(7, None, total_bytes = 100)["bytes_loaded"] == 80
    # Reads past the model files are clamped, and a full read reports ready.
    _patch_readers(monkeypatch, read = 120, rss = 5)
    snap = sampler.snapshot(7, None, total_bytes = 100)
    assert snap["bytes_loaded"] == 100
    assert snap["fraction"] == 1.0
    assert snap["phase"] == "ready"


def test_snapshot_returns_none_without_readable_counters(monkeypatch):
    sampler = sampler_module.LoadProgressSampler()
    _patch_readers(monkeypatch, read = None, rss = None)
    assert sampler.snapshot(7, None, total_bytes = 100) is None


def test_snapshot_without_known_total_reports_zero_fraction(monkeypatch):
    sampler = sampler_module.LoadProgressSampler()
    monkeypatch.setattr(sampler, "_vram_cached", lambda pid: None)
    _patch_readers(monkeypatch, read = 42, rss = 0)
    snap = sampler.snapshot(7, None, total_bytes = 0)
    assert snap["fraction"] == 0.0
    assert snap["phase"] == "loading"


def test_a_new_pid_does_not_inherit_the_previous_high_water(monkeypatch):
    sampler = sampler_module.LoadProgressSampler()
    monkeypatch.setattr(sampler, "_vram_cached", lambda pid: None)
    _patch_readers(monkeypatch, read = 80, rss = 0)
    assert sampler.snapshot(7, None, total_bytes = 100)["bytes_loaded"] == 80
    # The backend replaced sd-server and the OS reused the pid: a fresh load starts at 0.
    _patch_readers(monkeypatch, read = 5, rss = 0)
    snap = sampler.snapshot(7, None, total_bytes = 100)
    assert snap["bytes_loaded"] == 80  # same pid still keeps its mark
    assert sampler.snapshot(8, None, total_bytes = 100)["bytes_loaded"] == 5


def test_vram_query_runs_off_the_calling_thread(monkeypatch):
    sampler = sampler_module.LoadProgressSampler()
    started = threading.Event()
    release = threading.Event()
    seen = {}

    def fake_query(pid):
        seen["thread"] = threading.current_thread().name
        started.set()
        release.wait(2)
        return 999

    monkeypatch.setattr(sampler, "_query_vram", fake_query)
    _patch_readers(monkeypatch, read = 1, rss = 0)
    # A stale cache starts the refresh but snapshot() returns at once without a value.
    snap = sampler.snapshot(7, None, total_bytes = 100)
    assert snap["vram_mb"] is None
    assert started.wait(2)
    assert seen["thread"] != threading.current_thread().name
    release.set()
    for _ in range(200):
        if sampler._vram_value == 999:
            break
        time.sleep(0.01)
    assert sampler._vram_value == 999
    assert sampler.snapshot(7, None, total_bytes = 100)["vram_mb"] == 999
