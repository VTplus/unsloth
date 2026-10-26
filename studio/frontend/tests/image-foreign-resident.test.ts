// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  blocksLocalGenerate,
  canCompareOnResident,
  isForeignResident,
} from "../src/features/images/lib/foreign-resident.ts";

test("a hidden resident is foreign, even without a repo id", () => {
  assert.equal(isForeignResident({ loaded: true, yours: false }), true);
});

test("an own or absent resident is not foreign", () => {
  assert.equal(isForeignResident({ loaded: true, yours: true }), false);
  assert.equal(isForeignResident({ loaded: true }), false);
  assert.equal(isForeignResident({ loaded: false, yours: false }), false);
  assert.equal(isForeignResident(null), false);
  assert.equal(isForeignResident(undefined), false);
});

test("a foreign resident blocks local generation unless an external provider is picked", () => {
  const foreign = { loaded: true, yours: false };
  assert.equal(blocksLocalGenerate(foreign, false), true);
  assert.equal(blocksLocalGenerate(foreign, true), false);
  assert.equal(blocksLocalGenerate({ loaded: true, yours: true }, false), false);
  assert.equal(blocksLocalGenerate({ loaded: true }, false), false);
  assert.equal(blocksLocalGenerate({ loaded: false, yours: false }, false), false);
  assert.equal(blocksLocalGenerate(null, false), false);
});

test("the compare grid stays off for a foreign resident and outside create", () => {
  const foreign = { loaded: true, yours: false };
  const own = { loaded: true, yours: true };
  assert.equal(canCompareOnResident(foreign, false, "create"), false);
  assert.equal(canCompareOnResident(own, false, "create"), true);
  assert.equal(canCompareOnResident(own, true, "create"), false);
  assert.equal(canCompareOnResident(own, false, "edit"), false);
  assert.equal(canCompareOnResident({ loaded: false }, false, "create"), false);
  assert.equal(canCompareOnResident(undefined, false, "create"), false);
});
