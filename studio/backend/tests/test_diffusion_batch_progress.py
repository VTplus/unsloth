# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only arithmetic and clock tests for image job progress."""

import pytest

from core.inference import diffusion_batch_progress as bp


def test_serial_batch_has_local_steps_and_weighted_total(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: clock[0])
    p = bp.ImageBatchProgress(4)
    assert p.snapshot() is None
    p.begin(0, 1, 20)
    p.update(1)
    clock[0] += 2
    p.update(2)
    assert p.snapshot()["eta_seconds"] == 36
    p.update(20)
    assert p.snapshot()["fraction"] == 0.25
    p.begin(1, 1, 20)
    assert p.snapshot()["eta_seconds"] is None
    p.update(7)
    s = p.snapshot()
    assert (s["start"], s["end"], s["step"]) == (2, 2, 7)
    assert s["fraction"] == pytest.approx(1.35 / 4)


def test_parallel_groups_and_split_retry_use_image_jobs():
    p = bp.ImageBatchProgress(6)
    p.begin(0, 4, 8)
    p.update(4)
    s = p.snapshot()
    assert (s["start"], s["end"]) == (1, 4)
    assert s["fraction"] == pytest.approx(2 / 6)
    # The failed attempt returned no images: retry the same offset with half the group.
    p.begin(0, 2, 8)
    assert p.snapshot()["fraction"] == 0
    p.update(8)
    p.begin(2, 4, 8)
    p.update(8)
    assert p.snapshot()["fraction"] == 1


@pytest.mark.parametrize("offset,count,steps", [(-1, 1, 4), (0, 0, 4), (3, 2, 4), (0, 1, 0)])
def test_invalid_ranges_are_rejected(offset, count, steps):
    with pytest.raises(ValueError):
        bp.ImageBatchProgress(4).begin(offset, count, steps)


def test_duplicate_and_out_of_order_steps_do_not_rewind(monkeypatch):
    p = bp.ImageBatchProgress(2)
    p.begin(0, 1, 8)
    p.update(4)
    before = p.snapshot()
    monkeypatch.setattr(bp.time, "monotonic", lambda: pytest.fail("unchanged step read the clock"))
    p.update(4)
    p.update(1)
    assert p.snapshot() == before
    p.update(-1)
    assert p.snapshot() == before


def test_eta_uses_the_first_observed_step_even_if_callbacks_skip_steps(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: clock[0])
    p = bp.ImageBatchProgress(2)
    p.begin(0, 1, 10)
    p.update(5)
    assert p.snapshot()["eta_seconds"] is None
    clock[0] = 6.0
    p.update(7)
    assert p.snapshot()["eta_seconds"] == 9


def test_step_updates_clamp_into_the_group_range():
    p = bp.ImageBatchProgress(2)
    p.begin(0, 1, 4)
    p.update(9)
    assert p.snapshot()["step"] == 4
    p.update(-3)
    assert p.snapshot()["step"] == 4


def test_the_final_image_reports_the_full_batch_fraction():
    p = bp.ImageBatchProgress(2)
    p.begin(1, 1, 4)
    p.update(4)
    assert p.snapshot()["fraction"] == 1.0


def test_a_zero_total_batch_is_rejected():
    with pytest.raises(ValueError):
        bp.ImageBatchProgress(0)


def test_the_whole_batch_eta_anchors_at_the_first_observed_step(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: clock[0])
    p = bp.ImageBatchProgress(2)
    p.begin(0, 1, 4)
    p.update(2)
    assert p.snapshot()["total_eta_seconds"] is None
    clock[0] = 2.0
    p.update(4)
    assert p.snapshot()["total_eta_seconds"] == 2


def test_begin_resets_both_batch_clocks(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: clock[0])
    p = bp.ImageBatchProgress(4)
    p.begin(0, 1, 4)
    p.update(2)
    clock[0] = 2.0
    p.update(4)
    assert p.snapshot()["total_eta_seconds"] == 6
    p.begin(2, 1, 4)
    s = p.snapshot()
    assert s["total_eta_seconds"] is None
    assert s["eta_seconds"] is None
    assert s["fraction"] == 0.5


def test_the_final_image_completes_the_whole_batch_eta_at_zero(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(bp.time, "monotonic", lambda: clock[0])
    p = bp.ImageBatchProgress(2)
    p.begin(1, 1, 4)
    p.update(2)
    clock[0] = 2.0
    p.update(4)
    assert p.snapshot()["total_eta_seconds"] == 0
