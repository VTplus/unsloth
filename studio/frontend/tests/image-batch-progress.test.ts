// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  imageBatchTitle,
  imageStepProgress,
  runFraction,
} from "../src/features/images/lib/batch-progress.ts";
import { sameGenerateProgress, generatePhaseLabel } from "../src/lib/media-generate-phase.ts";

const batch = {
  total: 4, start: 2, end: 2, step: 7, total_steps: 20, fraction: 0.3375, eta_seconds: 26,
  total_eta_seconds: 210,
};
const p = { active: true, step: 27, total_steps: 80, fraction: 0.3375, eta_seconds: 106, batch };

test("image labels distinguish serial images from parallel groups", () => {
  assert.equal(imageBatchTitle(p), "Image 2/4");
  assert.equal(imageBatchTitle({ ...p, batch: { ...batch, start: 1, end: 4 } }), "Images 1–4/4");
  assert.equal(imageBatchTitle({ ...p, batch: { ...batch, total: 1, start: 1, end: 1 } }), null);
  assert.equal(imageBatchTitle(null), null);
});

test("the first bar uses local steps and ETA, keeping batch progress separate", () => {
  const local = imageStepProgress(p);
  assert.equal(local.step, 7);
  assert.equal(local.total_steps, 20);
  assert.equal(local.fraction, 0.35);
  assert.equal(local.eta_seconds, 26);
  assert.equal(local.batch?.fraction, 0.3375);
  assert.match(generatePhaseLabel({ ...local, total: local.total_steps }), /Step 7\/20 · ~26s$/);
  assert.equal(generatePhaseLabel({ ...local, total: local.total_steps, phase: "decode" }), "Decoding…");
});

test("old engines and connections retain the existing progress", () => {
  const old = { ...p, batch: undefined };
  assert.equal(imageStepProgress(old), old);
  assert.equal(imageBatchTitle(old), null);
});

test("polling notices a new image even at the same step and ETA", () => {
  assert.equal(sameGenerateProgress(p, { ...p, batch: { ...batch } }), true);
  for (const delta of [
    { start: 3, end: 3 }, { total: 5 }, { end: 3 }, { step: 8 },
    { total_steps: 10 }, { fraction: 0.5 }, { eta_seconds: null },
    { total_eta_seconds: null },
  ]) assert.equal(sameGenerateProgress(p, { ...p, batch: { ...batch, ...delta } }), false);
  assert.equal(sameGenerateProgress(p, { ...p, total_steps: 81 }), false);
  assert.equal(sameGenerateProgress(p, { ...p, batch: null }), false);
});

test("run progress aggregates finished runs with the live batch share", () => {
  assert.equal(runFraction(1, 0, p), null);
  assert.equal(runFraction(4, null, p), null);
  assert.equal(runFraction(4, 1, { ...p, batch: { ...batch, fraction: 0.5 } }), 0.375);
  assert.equal(runFraction(4, 2, { ...p, fraction: 0.5, batch: null }), 0.625);
  assert.equal(runFraction(4, 2, null), 0.5);
  assert.equal(runFraction(4, 3, { ...p, batch: { ...batch, fraction: 1 } }), 1);
});
