// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ResidentStatus = { loaded?: boolean; yours?: boolean } | null | undefined;

/** A resident that exists but belongs to another account: `{loaded: true, yours: false}`, no repo_id. */
export function isForeignResident(status: ResidentStatus): boolean {
  return Boolean(status?.loaded && status.yours === false);
}

/** A local generate must not run on another account's resident; an external provider never touches it.
 *  ``external`` is the picked external model (truthy) or null; only its truthiness matters. */
export function blocksLocalGenerate(status: ResidentStatus, external: unknown): boolean {
  return !external && isForeignResident(status);
}

/** The Compare grid calls generate directly, so it needs the same gate as the main button. */
export function canCompareOnResident(
  status: ResidentStatus,
  external: unknown,
  workflow: string,
): boolean {
  return (
    !external && !isForeignResident(status) && Boolean(status?.loaded) && workflow === "create"
  );
}

/** Shown by the toast and the footer, so the wording cannot drift between them. */
export const FOREIGN_RESIDENT_NOTICE =
  "This image model is loaded by another account. Pick a model to load it for this account.";
