// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface MediaGeneratePhaseProgress {
  // enhance (FLUX.3 prompt upsampler) | encode | denoise | decode | export (video); absent or null reads as
  // denoise.
  phase?: string | null;
  step: number;
  total: number;
  eta_seconds?: number | null;
}

export function generatePhaseLabel(
  p: MediaGeneratePhaseProgress,
  opts: { hasAudio?: boolean; formatEta?: (seconds: number) => string } = {},
): string {
  if (p.phase === "enhance") return "Enhancing prompt…";
  if (p.phase === "encode") return "Encoding prompt…";
  if (p.phase === "decode") return opts.hasAudio ? "Decoding video and audio…" : "Decoding…";
  if (p.phase === "export") return "Encoding video…";
  const base = "Denoising please wait…";
  // Step 0 is still running (compile warmup included), so no 0/N is shown.
  if (p.step <= 0 || p.total <= 0) return base;
  const fmt = opts.formatEta ?? ((seconds: number) => `${Math.max(0, Math.round(seconds))}s`);
  const eta = p.eta_seconds != null ? fmt(p.eta_seconds) : "";
  // Bullet separator matches ModelLoadDescription's line split; one line truncated in the 18rem card.
  return `${base} \u2022 Step ${p.step}/${p.total}${eta ? ` · ~${eta}` : ""}`;
}

interface ComparableGenerateProgress {
  step: number;
  total_steps?: number;
  fraction?: number;
  eta_seconds?: number | null;
  phase?: string | null;
  preview_seq?: number;
  batch?: {
    total: number;
    start: number;
    end: number;
    step: number;
    total_steps: number;
    fraction: number;
    eta_seconds: number | null;
    total_eta_seconds: number | null;
  } | null;
}

export function sameGenerateProgress(
  a: ComparableGenerateProgress,
  b: ComparableGenerateProgress,
): boolean {
  return (
    a.step === b.step &&
    a.total_steps === b.total_steps &&
    a.fraction === b.fraction &&
    a.batch?.total === b.batch?.total &&
    a.batch?.start === b.batch?.start &&
    a.batch?.end === b.batch?.end &&
    a.batch?.step === b.batch?.step &&
    a.batch?.total_steps === b.batch?.total_steps &&
    a.batch?.fraction === b.batch?.fraction &&
    a.batch?.eta_seconds === b.batch?.eta_seconds &&
    a.batch?.total_eta_seconds === b.batch?.total_eta_seconds &&
    a.eta_seconds === b.eta_seconds &&
    a.phase === b.phase &&
    (a.preview_seq ?? 0) === (b.preview_seq ?? 0)
  );
}
