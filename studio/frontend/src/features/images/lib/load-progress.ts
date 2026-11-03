// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionGenerateProgress, DiffusionModelLoadProgress } from "../api";

/** The weight-load object while the model files are still being read; null once their bytes are in. */
export function activeLoadProgress(
  progress: DiffusionGenerateProgress | null | undefined,
): DiffusionModelLoadProgress | null {
  const load = progress?.load;
  if (!load || load.phase === "ready" || load.fraction >= 1) return null;
  return load;
}

/** Ring fraction for the Stop button, clamped to [0,1]. */
export function loadRingFraction(
  load: DiffusionModelLoadProgress | null | undefined,
): number | null {
  if (!load || load.phase === "ready" || load.fraction >= 1) return null;
  return Math.max(0, Math.min(1, load.fraction));
}

/** Overlay progress value while the weights are read; null once they are in. */
export function loadProgressPercent(
  load: DiffusionModelLoadProgress | null | undefined,
): number | null {
  const fraction = loadRingFraction(load);
  return fraction == null ? null : fraction * 100;
}

/** Overlay title while the weights are read; null once they are in. */
export function loadProgressTitle(
  load: DiffusionModelLoadProgress | null | undefined,
): string | null {
  return loadRingFraction(load) == null ? null : "Loading model";
}

/** Overlay label; the VRAM value rides as the secondary line (" • " splits in ModelLoadDescription). */
export function loadProgressLabel(
  load: DiffusionModelLoadProgress | null | undefined,
): string | null {
  if (loadRingFraction(load) == null) return null;
  const vram = load?.vram_mb;
  return vram && vram > 0
    ? `Loading model • ${vram.toLocaleString("en-US")} MB VRAM`
    : "Loading model";
}
