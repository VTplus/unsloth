// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionModelLoadProgress } from "../api";

/** Fraction of the model files read, clamped to [0,1]. */
export function loadFraction(
  load: DiffusionModelLoadProgress | null | undefined,
): number | null {
  if (!load || load.phase === "ready" || load.fraction >= 1) return null;
  return Math.max(0, Math.min(1, load.fraction));
}

/** Overlay progress value while the weights are read; null once they are in. */
export function loadProgressPercent(
  load: DiffusionModelLoadProgress | null | undefined,
): number | null {
  const fraction = loadFraction(load);
  return fraction == null ? null : fraction * 100;
}

/** Overlay title while the weights are read; null once they are in. */
export function loadProgressTitle(
  load: DiffusionModelLoadProgress | null | undefined,
): string | null {
  return loadFraction(load) == null ? null : "Loading model";
}

/** Overlay label; the VRAM value is the primary label line. */
export function loadProgressLabel(
  load: DiffusionModelLoadProgress | null | undefined,
): string | null {
  if (loadFraction(load) == null) return null;
  const vram = load?.vram_mb;
  return vram && vram > 0 ? `${(vram / 1024).toFixed(2)} GiB VRAM` : null;
}

/** Overlay label; the step label only applies once no load is active. */
export function loadProgressFallbackLabel(
  load: DiffusionModelLoadProgress | null | undefined,
  stepLabel: string | null,
): string | null {
  return loadProgressLabel(load) ?? (loadProgressTitle(load) == null ? stepLabel : null);
}
