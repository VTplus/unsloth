// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionGenerateProgress } from "../api";

/** Preserve the older engine/connection response when no per-image sampling data is available. */
export function imageStepProgress(p: DiffusionGenerateProgress): DiffusionGenerateProgress {
  const b = p.batch;
  if (!b || b.total <= 1) return p;
  return {
    ...p,
    step: b.step,
    total_steps: b.total_steps,
    fraction: b.step / b.total_steps,
    eta_seconds: b.eta_seconds,
  };
}

export function imageBatchTitle(p: DiffusionGenerateProgress | null): string | null {
  const b = p?.batch;
  if (!b || b.total <= 1) return null;
  return b.start === b.end
    ? `Image ${b.start}/${b.total}`
    : `Images ${b.start}–${b.end}/${b.total}`;
}

/** Aggregate run progress for the Count loop: finished runs plus the live run's denoising share. */
export function runFraction(
  count: number,
  genDone: number | null | undefined,
  genStep: DiffusionGenerateProgress | null | undefined,
): number | null {
  if (count <= 1 || genDone == null) return null;
  const b = genStep?.batch;
  const share = b ? b.fraction : (genStep?.fraction ?? 0);
  return Math.min(1, (genDone + share) / count);
}
