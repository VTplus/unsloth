// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type {
  DiffusionGenerateProgress,
  DiffusionModelLoadProgress,
} from "../src/features/images/api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { activeLoadProgress, loadRingFraction, loadProgressLabel } = await import(
  "../src/features/images/lib/load-progress.ts"
);

const load = (over: Partial<DiffusionModelLoadProgress> = {}): DiffusionModelLoadProgress => ({
  phase: "loading",
  bytes_loaded: 5,
  bytes_total: 10,
  fraction: 0.5,
  vram_mb: 18292,
  eta_seconds: null,
  ...over,
});

const progress = (
  over: Partial<DiffusionGenerateProgress> = {},
): DiffusionGenerateProgress => ({
  active: true,
  step: 0,
  total_steps: 10,
  fraction: 0,
  eta_seconds: null,
  ...over,
});

test("ring fraction follows the read share and disappears when the load is done", () => {
  assert.equal(loadRingFraction(load()), 0.5);
  assert.equal(loadRingFraction(load({ phase: "ready" })), null);
  assert.equal(loadRingFraction(load({ fraction: 1 })), null);
  assert.equal(loadRingFraction(null), null);
});

test("active load only exists while the weights are still being read", () => {
  assert.equal(activeLoadProgress(progress({ load: load() }))?.fraction, 0.5);
  assert.equal(activeLoadProgress(progress({ load: load({ phase: "ready" }) })), null);
  assert.equal(activeLoadProgress(progress({ load: load({ fraction: 1 }) })), null);
  assert.equal(activeLoadProgress(null), null);
});

test("label carries the VRAM value as the secondary line", () => {
  assert.equal(loadProgressLabel(load()), "Loading model • 18,292 MB VRAM");
  assert.equal(loadProgressLabel(load({ vram_mb: null })), "Loading model");
  assert.equal(loadProgressLabel(load({ phase: "ready" })), null);
});
