# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-image/group sampling progress, independent of torch and of the image engine."""

from __future__ import annotations

import time
from typing import Optional


class ImageBatchProgress:
    def __init__(self, total: int):
        if total < 1:
            raise ValueError("An image batch must contain at least one job")
        self.total = total
        # Replace one tuple so a concurrent poll cannot mix two images' counters. Slots:
        # offset, count, step, total_steps, first_step, first, eta, anchor_fraction, anchor_at, total_eta.
        self._state: Optional[
            tuple[
                int,
                int,
                int,
                int,
                int,
                Optional[float],
                Optional[float],
                float,
                Optional[float],
                Optional[float],
            ]
        ] = None

    def begin(self, offset: int, count: int, total_steps: int) -> None:
        if offset < 0 or count < 1 or offset + count > self.total or total_steps < 1:
            raise ValueError("Invalid image batch range or step count")
        self._state = (
            offset,
            count,
            0,
            total_steps,
            0,
            None,
            None,
            offset / self.total,
            None,
            None,
        )

    def update(self, step: int) -> None:
        state = self._state
        if state is None:
            return
        (
            offset,
            count,
            previous,
            total_steps,
            first_step,
            first,
            eta,
            anchor_fraction,
            anchor_at,
            total_eta,
        ) = state
        step = max(previous, min(max(0, int(step)), total_steps))
        if step == previous:
            return
        now = time.monotonic()
        if first is None:
            # Both clocks anchor at the first observed step, not at begin(), so encode and
            # setup time stay out of the extrapolations.
            first = now
            first_step = step
            anchor_at = now
        else:
            if step > first_step:
                eta = max(0.0, (total_steps - step) * (now - first) / (step - first_step))
            fraction = min(1.0, (offset + count * step / total_steps) / self.total)
            if fraction > anchor_fraction:
                total_eta = max(
                    0.0, (1.0 - fraction) * (now - anchor_at) / (fraction - anchor_fraction)
                )
        self._state = (
            offset,
            count,
            step,
            total_steps,
            first_step,
            first,
            eta,
            anchor_fraction,
            anchor_at,
            total_eta,
        )

    def snapshot(self) -> Optional[dict]:
        state = self._state
        if state is None:
            return None
        offset, count, step, total_steps, _, _, eta, _, _, total_eta = state
        return {
            "total": self.total,
            "start": offset + 1,
            "end": offset + count,
            "step": step,
            "total_steps": total_steps,
            "fraction": min(1.0, (offset + count * step / total_steps) / self.total),
            "eta_seconds": eta,
            "total_eta_seconds": total_eta,
        }
