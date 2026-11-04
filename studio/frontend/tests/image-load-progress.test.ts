// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { DiffusionModelLoadProgress } from "../src/features/images/api.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  loadFraction,
  loadProgressFallbackLabel,
  loadProgressLabel,
  loadProgressPercent,
  loadProgressTitle,
} = await import("../src/features/images/lib/load-progress.ts");

const load = (over: Partial<DiffusionModelLoadProgress> = {}): DiffusionModelLoadProgress => ({
  phase: "loading",
  bytes_loaded: 5,
  bytes_total: 10,
  fraction: 0.5,
  vram_mb: 18292,
  eta_seconds: null,
  ...over,
});

test("load fraction follows the read share and disappears when the load is done", () => {
  assert.equal(loadFraction(load()), 0.5);
  assert.equal(loadFraction(load({ phase: "ready" })), null);
  assert.equal(loadFraction(load({ fraction: 1 })), null);
  assert.equal(loadFraction(null), null);
});

test("label shows the VRAM value in GiB and hides without it", () => {
  assert.equal(loadProgressLabel(load()), "17.86 GiB VRAM");
  assert.equal(loadProgressLabel(load({ vram_mb: null })), null);
  assert.equal(loadProgressLabel(load({ phase: "ready" })), null);
});

test("overlay helpers derive percent and title from the active load", () => {
  assert.equal(loadProgressPercent(load()), 50);
  assert.equal(loadProgressPercent(load({ fraction: 1 })), null);
  assert.equal(loadProgressTitle(load()), "Loading model");
  assert.equal(loadProgressTitle(load({ phase: "ready" })), null);
  assert.equal(loadProgressTitle(null), null);
});

test("fallback label only uses the step label when no load is active", () => {
  assert.equal(loadProgressFallbackLabel(load(), "Denoising please wait…"), "17.86 GiB VRAM");
  assert.equal(loadProgressFallbackLabel(load({ vram_mb: null }), "Denoising please wait…"), null);
  assert.equal(
    loadProgressFallbackLabel(load({ phase: "ready" }), "Denoising please wait…"),
    "Denoising please wait…",
  );
});
