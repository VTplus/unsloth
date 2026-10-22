# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small-host load route (``diffusion_small_host.py``): host-RAM pre-check, decision table, the memory-mapped
conversions, and the plan the route pins.

Sizes are the ones a Kaggle T4 (fp16 only, 15 GB, 31 GB host RAM) sees: FLUX.1-schnell stores a 22.7 GB bf16
transformer and an 8.9 GB bf16 T5; Qwen-Image a 38.9 GB transformer and a 15.8 GB Qwen2.5-VL. CPU only.
"""

from __future__ import annotations

import json
import struct
import sys
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_small_host as sh
from .test_diffusion_backend import fake_runtime  # noqa: F401 -- reused CPU-only loader fixture

FLUX1 = {
    "transformer": sh.StoredComponent("transformer", 22700, "bfloat16"),
    "text_encoder_2": sh.StoredComponent("text_encoder_2", 9083, "bfloat16"),
    "text_encoder": sh.StoredComponent("text_encoder", 235, "float32"),
    "vae": sh.StoredComponent("vae", 160, "float32"),
}
QWEN = {
    "transformer": sh.StoredComponent("transformer", 38900, "bfloat16"),
    "text_encoder": sh.StoredComponent("text_encoder", 15820, "bfloat16"),
    "vae": sh.StoredComponent("vae", 242, "bfloat16"),
}
KAGGLE_TOTAL = 31 * 1024
KAGGLE_AVAILABLE = 29900


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    for name in (sh.SMALL_HOST_ENV, sh.HOST_RAM_CHECK_ENV, sh.HOST_RAM_LIMIT_ENV):
        monkeypatch.delenv(name, raising = False)


def _decide(
    comps,
    dtype = torch.float16,
    total = KAGGLE_TOTAL,
    available = KAGGLE_AVAILABLE,
    **kw,
):
    return sh.decide_small_host(
        comps,
        dtype,
        device = kw.pop("device", "cuda"),
        host_total_mib = total,
        host_available_mib = available,
        **kw,
    )


def test_flux1_on_a_31gb_fp16_host_takes_the_route():
    d = _decide(FLUX1)
    assert d.engaged and d.refuse is None
    # only the large bf16 components load memory-mapped; the fp32 CLIP and VAE convert as before
    assert d.storage_dtypes == {"transformer": "bfloat16", "text_encoder_2": "bfloat16"}
    assert d.dense_host_mib > KAGGLE_AVAILABLE
    assert d.route_host_mib == 22700 // 2


def test_unet_and_large_vae_are_budgeted_at_their_converted_size():
    comps = {
        "unet": sh.StoredComponent("unet", 4900, "bfloat16"),
        "vae": sh.StoredComponent("vae", 600, "bfloat16"),
        "text_encoder_2": sh.StoredComponent("text_encoder_2", 1300, "bfloat16"),
    }
    d = _decide(comps, total = 8 * 1024, available = 6000)
    assert d.engaged
    # only a DiT is stored as int8; the UNet and VAE convert dense on the host, the encoder stays memory-mapped
    assert d.route_host_mib == 4900 + 600
    assert d.refuse is not None
    assert (
        d.route_host_mib
        == _decide(comps, dtype = torch.float32, total = 8 * 1024, available = 6000).route_host_mib // 2
    )


def test_qwen_image_fp32_promoted_takes_the_route():
    d = _decide(QWEN, dtype = torch.float32)
    assert d.engaged and d.refuse is None
    assert d.dense_host_mib > 2 * (38900 + 15820)


def test_large_host_and_bf16_cards_are_untouched():
    assert not _decide(FLUX1, total = 230 * 1024, available = 220 * 1024).engaged
    # bf16 compute loads a bf16 repo memory-mapped already: never routed, whatever the host
    assert not _decide(FLUX1, dtype = torch.bfloat16).engaged
    assert not _decide(FLUX1, device = "mps").engaged
    assert not _decide({"transformer": sh.StoredComponent("transformer", 9000, "float16")}).engaged


def test_kill_switch_and_force(monkeypatch):
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "0")
    assert not _decide(FLUX1).engaged
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    assert _decide(FLUX1, total = 230 * 1024, available = 220 * 1024).engaged


def test_refuses_before_loading_when_even_the_route_cannot_fit(monkeypatch):
    d = _decide(QWEN, dtype = torch.float32, total = 16 * 1024, available = 14000)
    assert d.engaged and d.refuse and "Not enough system RAM" in d.refuse
    monkeypatch.setenv(sh.HOST_RAM_CHECK_ENV, "0")
    assert _decide(QWEN, dtype = torch.float32, total = 16 * 1024, available = 14000).refuse is None


def test_flux1_dense_load_fits_the_physical_budget():
    decision = _decide(FLUX1, total = 65536, available = 59392)
    assert decision.dense_host_mib == 36550
    assert sh.host_reserve_mib(65536) == 6553
    assert decision.dense_host_mib + sh.host_reserve_mib(65536) == 43103
    assert not decision.engaged
    assert decision.refuse is None
    assert decision.storage_dtypes == {}


@pytest.mark.parametrize("prebuilt", [0, 4096])
def test_prebuilt_diagnostic_does_not_force_the_qwen_route(prebuilt):
    components = {
        "text_encoder": sh.StoredComponent("text_encoder", 16717, "bfloat16"),
        "vae": sh.StoredComponent("vae", 335, "bfloat16"),
    }
    decision = _decide(components, total = 48128, available = 36864, prebuilt_host_mib = prebuilt)
    assert decision.dense_host_mib == 19224
    assert sh.host_reserve_mib(48128) == 4812
    assert decision.prebuilt_host_mib == prebuilt
    assert not decision.engaged
    assert decision.refuse is None
    assert decision.storage_dtypes == {}


@pytest.mark.parametrize("prebuilt", [0, 4096])
def test_prebuilt_diagnostic_does_not_refuse_the_qwen_route(prebuilt):
    components = {
        "text_encoder": sh.StoredComponent("text_encoder", 16717, "bfloat16"),
        "vae": sh.StoredComponent("vae", 335, "bfloat16"),
    }
    decision = _decide(components, total = 48128, available = 7500, prebuilt_host_mib = prebuilt)
    assert decision.dense_host_mib == 19224
    assert decision.route_host_mib == 0
    assert sh.host_reserve_mib(48128) == 4812
    assert decision.prebuilt_host_mib == prebuilt
    assert decision.engaged
    assert decision.refuse is None
    assert decision.storage_dtypes == {"text_encoder": "bfloat16"}


def _qwen_companions(source_mib = 16717):
    return {
        "text_encoder": sh.StoredComponent(
            "text_encoder", 16717, "bfloat16", source_mib = source_mib
        ),
        "vae": sh.StoredComponent("vae", 335, "bfloat16"),
    }


def _commit_decide(**overrides):
    kwargs = dict(
        total = 48128,
        available = 36864,
        host_commit_mib = 30720,
        dense_workspace_mib = 0,
        route_workspace_mib = 0,
    )
    kwargs.update(overrides)
    components = kwargs.pop("components", None)
    return _decide(_qwen_companions() if components is None else components, **kwargs)


def test_commit_routes_a_load_whose_physical_budget_fits():
    decision = _commit_decide()
    assert decision.dense_host_mib == 19224
    assert decision.route_host_mib == 0
    assert decision.reserve_mib == 4812
    assert decision.source_mapping_mib == 16717
    assert decision.dense_commit_mib == 35941
    assert decision.route_commit_mib == 16717
    assert decision.dense_commit_mib + decision.reserve_mib == 40753
    assert decision.route_commit_mib + decision.reserve_mib == 21529
    assert decision.commit_available_mib == 30720
    assert decision.engaged and decision.refuse is None
    assert "commit" in decision.reason.lower()


def test_commit_refuses_when_the_route_also_exceeds_headroom():
    decision = _commit_decide(host_commit_mib = 20480)
    assert decision.engaged and decision.refuse is not None
    assert decision.route_commit_mib + decision.reserve_mib == 21529
    assert "commit" in decision.refuse.lower()


def test_physical_shortfall_has_a_physical_reason():
    decision = _commit_decide(available = 16000, host_commit_mib = 60000)
    assert decision.engaged and decision.refuse is None
    assert "physical" in decision.reason.lower()


def test_ample_commit_does_not_quantize_a_physical_fits_load():
    decision = _commit_decide(host_commit_mib = 60000)
    assert not decision.engaged and decision.refuse is None
    assert decision.storage_dtypes == {}
    assert sh.torch_dtype_map(decision, torch.float16) is torch.float16


@pytest.mark.parametrize("source_mib", [None, 16717])
@pytest.mark.parametrize("workspace_mib", [None, 0])
def test_zero_commit_is_exhausted_even_with_unknown_estimates(source_mib, workspace_mib):
    decision = _commit_decide(
        components = _qwen_companions(source_mib),
        host_commit_mib = 0,
        dense_workspace_mib = workspace_mib,
        route_workspace_mib = workspace_mib,
    )
    assert decision.engaged and decision.refuse is not None
    assert decision.commit_available_mib == 0 and decision.reserve_mib == 4812
    assert "commit" in decision.refuse.lower()
    if source_mib is None or workspace_mib is None:
        assert decision.dense_commit_mib is None and decision.route_commit_mib is None


@pytest.mark.parametrize("commit", [None, "raises"])
def test_unavailable_commit_reader_preserves_physical_budget(monkeypatch, commit):
    calls = []

    def read(name, value):
        calls.append(name)
        if value == "raises":
            raise OSError("optional reader unavailable")
        return value

    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: read("total", 48128))
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: read("physical", 36864))
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: read("commit", commit))
    assert dm._host_memory_budgets_mib() == (48128, 36864, None)
    assert calls == ["total", "physical", "commit"]
    decision = _commit_decide(host_commit_mib = None)
    assert not decision.engaged and decision.refuse is None


@pytest.mark.parametrize("commit,refuses", [(30720, False), (20480, True)])
def test_commit_still_binds_when_physical_available_is_unknown(commit, refuses):
    decision = _commit_decide(available = None, host_commit_mib = commit)
    assert decision.available_mib is None and decision.commit_available_mib == commit
    assert decision.engaged
    assert (decision.refuse is not None) is refuses
    assert "commit" in decision.reason.lower()


def test_both_unknown_budgets_keep_the_unmeasured_return():
    decision = _commit_decide(available = None, host_commit_mib = None)
    assert not decision.engaged and decision.refuse is None
    assert decision.available_mib is None and decision.commit_available_mib is None
    assert "unreadable" in decision.reason.lower()


@pytest.mark.parametrize(
    "boundary,available,commit,equal_engaged",
    [
        ("dense physical", 24036, 60000, False),
        ("route physical", 4812, 60000, True),
        ("dense commit", 36864, 40753, False),
        ("route commit", 36864, 21529, True),
    ],
)
def test_exact_budget_boundary_fits_and_one_less_does_not(
    boundary, available, commit, equal_engaged
):
    equal = _commit_decide(available = available, host_commit_mib = commit)
    assert equal.engaged is equal_engaged and equal.refuse is None
    below = _commit_decide(
        available = available - int("physical" in boundary),
        host_commit_mib = commit - int("commit" in boundary),
    )
    assert below.engaged
    assert (below.refuse is not None) is boundary.startswith("route")


@pytest.mark.parametrize("dtype,dense", [(torch.float16, 1177), (torch.float32, 2355)])
def test_sources_and_kept_fp32_targets_are_budgeted_separately(dtype, dense):
    components = {
        "text_encoder": sh.StoredComponent("text_encoder", 1024, "bfloat16", 128, 1025),
    }
    decision = _commit_decide(components = components, dtype = dtype, available = 50000)
    assert decision.dense_host_mib == dense
    assert decision.route_host_mib == 256
    assert decision.source_mapping_mib == 1025
    assert decision.dense_commit_mib == 1025 + dense
    assert decision.route_commit_mib == 1281
    assert decision.dense_commit_mib != 2 * decision.dense_host_mib


def test_unknown_estimates_preserve_known_commit_lower_bounds():
    missing_source = _commit_decide(components = _qwen_companions(None), host_commit_mib = 60000)
    assert not missing_source.engaged and missing_source.refuse is None
    assert missing_source.source_mapping_mib is None
    assert missing_source.dense_commit_mib is None and missing_source.route_commit_mib is None
    assert missing_source.dense_commit_lower_bound_mib == 19224
    assert missing_source.route_commit_lower_bound_mib == 0
    missing_workspace = _commit_decide(
        host_commit_mib = 21528, dense_workspace_mib = None, route_workspace_mib = None
    )
    assert missing_workspace.engaged and missing_workspace.refuse is not None
    assert missing_workspace.dense_commit_mib is None and missing_workspace.route_commit_mib is None
    assert missing_workspace.dense_commit_lower_bound_mib == 35941
    assert missing_workspace.route_commit_lower_bound_mib == 16717
    assert "commit" in missing_workspace.refuse.lower()


def test_workspace_only_increases_the_commit_budget():
    baseline = _commit_decide(host_commit_mib = 40753)
    workspace = _commit_decide(host_commit_mib = 40753, dense_workspace_mib = 1024)
    assert not baseline.engaged and workspace.engaged and workspace.refuse is None
    assert workspace.dense_commit_mib == baseline.dense_commit_mib + 1024
    assert workspace.dense_host_mib == baseline.dense_host_mib == 19224
    assert workspace.route_host_mib == baseline.route_host_mib == 0
    assert workspace.source_mapping_mib == baseline.source_mapping_mib == 16717
    for extra in (0, 1024):
        physical_only = _commit_decide(host_commit_mib = None, dense_workspace_mib = extra)
        assert not physical_only.engaged and physical_only.refuse is None


@pytest.mark.parametrize("cap", ["6144", "90000", "invalid", "0", "-1", ""])
@pytest.mark.parametrize(
    "budgets",
    [
        (32000, 20000, 12000),
        (None, 0, None),
        (8192, 1024, 40000),
        (None, 20000, 12000),
        (32000, None, 12000),
        (32000, 20000, None),
        (None, None, None),
    ],
)
def test_cap_only_lowers_known_physical_and_commit_budgets(monkeypatch, cap, budgets):
    monkeypatch.setattr(dm, "_host_memory_budgets_mib", lambda: budgets)
    monkeypatch.setenv(sh.HOST_RAM_LIMIT_ENV, cap)
    limit = int(cap) if cap.lstrip("-").isdigit() else 0
    expected = tuple(
        min(value, limit) if value is not None and limit > 0 else value for value in budgets
    )
    assert sh.host_memory_budgets_mib() == expected
    assert sh.host_reserve_mib(expected[0]) == max(3072, int((expected[0] or 0) * 0.1))


def test_kill_switch_force_and_refusal_override_keep_their_contract(monkeypatch):
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "0")
    disabled = _commit_decide(host_commit_mib = 0)
    assert not disabled.engaged and disabled.refuse is None
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    forced = _commit_decide(host_commit_mib = 0)
    assert forced.engaged and forced.refuse is not None
    monkeypatch.setenv(sh.HOST_RAM_CHECK_ENV, "0")
    unchecked = _commit_decide(host_commit_mib = 0)
    assert unchecked.engaged and unchecked.refuse is None
    monkeypatch.delenv(sh.SMALL_HOST_ENV)
    still_routes = _commit_decide(host_commit_mib = 0)
    assert still_routes.engaged and still_routes.refuse is None


@pytest.mark.parametrize("forced", [False, True])
@pytest.mark.parametrize("prebuilt", [0, 4096])
def test_lora_override_and_force_preserve_diagnostic_metadata(monkeypatch, forced, prebuilt):
    if forced:
        monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    decision = _commit_decide(host_commit_mib = 0, lora_active = True, prebuilt_host_mib = prebuilt)
    assert decision.engaged is forced
    assert (decision.refuse is not None) is forced
    assert decision.prebuilt_host_mib == prebuilt
    assert decision.dense_host_mib == 19224 and decision.route_host_mib == 0
    assert decision.source_mapping_mib == 16717
    if not forced:
        assert "LoRA" in decision.reason


@pytest.mark.parametrize(
    "device,dtype,components",
    [
        ("cpu", torch.float16, FLUX1),
        ("mps", torch.float16, FLUX1),
        ("cuda", torch.bfloat16, FLUX1),
        ("cuda", torch.float16, {"vae": sh.StoredComponent("vae", 511, "bfloat16")}),
    ],
)
def test_force_does_not_expand_existing_device_dtype_or_size_gates(
    monkeypatch, device, dtype, components
):
    monkeypatch.setenv(sh.SMALL_HOST_ENV, "1")
    decision = _decide(components, device = device, dtype = dtype, host_commit_mib = 0)
    assert not decision.engaged and decision.refuse is None


def test_unresolved_snapshot_keeps_the_existing_early_return(monkeypatch):
    from core.inference import diffusion as dif

    monkeypatch.setattr(dif, "resolve_snapshot_dir", lambda *args: None)
    monkeypatch.setattr(
        dif, "host_memory_budgets_mib", lambda: pytest.fail("snapshot gate must return first")
    )
    assert (
        dif.DiffusionBackend()._small_host_decision(
            "unresolved/repo",
            {},
            types.SimpleNamespace(device = "cuda"),
            torch.float16,
            lora_active = False,
        )
        is None
    )


@pytest.mark.parametrize(
    ("lora_active", "available"),
    [(False, 100 * 1024), (True, KAGGLE_AVAILABLE)],
)
def test_dense_decisions_report_prebuilt_host_bytes(lora_active, available):
    baseline = _decide(FLUX1, total = 128 * 1024, available = available, lora_active = lora_active)
    decision = _decide(
        FLUX1,
        total = 128 * 1024,
        available = available,
        lora_active = lora_active,
        prebuilt_host_mib = 4096,
    )
    assert not decision.engaged and decision.refuse is None
    assert decision.prebuilt_host_mib == 4096
    assert decision.dense_host_mib == baseline.dense_host_mib
    assert decision.route_host_mib == baseline.route_host_mib
    assert decision.storage_dtypes == baseline.storage_dtypes
    assert ("LoRA" in decision.reason) is lora_active


def test_host_ram_limit_env_caps_the_guard_view(monkeypatch):
    monkeypatch.setattr(dm, "_host_memory_budgets_mib", lambda: (32000, 20000, 24000))
    monkeypatch.setenv(sh.HOST_RAM_LIMIT_ENV, "6144")
    assert sh.host_ram_mib() == (6144, 6144)
    monkeypatch.setenv(sh.HOST_RAM_LIMIT_ENV, "90000")
    assert sh.host_ram_mib() == (32000, 20000)
    monkeypatch.setenv(sh.HOST_RAM_LIMIT_ENV, "not-a-number")
    assert sh.host_ram_mib() == (32000, 20000)


def test_separate_budgets_do_not_collapse_physical_and_commit(monkeypatch):
    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: 16000)
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 5000)
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: 3000)
    assert dm._host_memory_budgets_mib() == (16000, 5000, 3000)
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: 9000)
    assert dm._host_memory_budgets_mib() == (16000, 5000, 9000)
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: None)
    assert dm._host_memory_budgets_mib() == (16000, 5000, None)


def test_windows_commit_headroom_reader_reports_a_value_on_windows():
    # the reader once called an unexported kernel32 name and silently returned None, leaving the guard on the
    # physical reading; on Windows it must reach the real API, elsewhere None is by design
    if sys.platform != "win32":
        pytest.skip("Windows-only reader")
    value = dm._windows_commit_headroom_mib()
    assert value is not None and value >= 0


@pytest.mark.parametrize(
    "outcome,expected", [("ok", 8), ("exhausted", 0), ("failure", None), ("raises", None)]
)
def test_windows_reader_distinguishes_exhaustion_from_api_failure(monkeypatch, outcome, expected):
    import ctypes

    calls = []

    def performance_info(pointer, size):
        calls.append("K32GetPerformanceInfo")
        if outcome == "raises":
            raise OSError("optional API unavailable")
        info = pointer._obj
        assert size == ctypes.sizeof(info) and info.cb == size
        info.CommitLimit = 4096
        info.CommitTotal = 4096 if outcome == "exhausted" else 2048
        info.PageSize = 4096
        return int(outcome != "failure")

    def library(name, **kwargs):
        assert name == "kernel32" and kwargs == {"use_last_error": True}
        return types.SimpleNamespace(K32GetPerformanceInfo = performance_info)

    monkeypatch.setattr(dm.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", library, raising = False)
    assert dm._windows_commit_headroom_mib() == expected
    assert calls == ["K32GetPerformanceInfo"]


@pytest.mark.parametrize(
    "available,commit,lora,engaged,refuses,dense_check,route_check",
    [
        (
            36864,
            60000,
            False,
            False,
            False,
            "physical 19224 + reserve 4812 = 24036 MiB <= 36864 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 35941 + reserve 4812 = 40753 MiB <= 60000 MiB",
            None,
        ),
        (
            16000,
            60000,
            False,
            True,
            False,
            "physical 19224 + reserve 4812 = 24036 MiB > 16000 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 35941 + reserve 4812 = 40753 MiB <= 60000 MiB",
            "physical 0 + reserve 4812 = 4812 MiB <= 16000 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 16717 + reserve 4812 = 21529 MiB <= 60000 MiB",
        ),
        (
            36864,
            20480,
            False,
            True,
            True,
            "physical 19224 + reserve 4812 = 24036 MiB <= 36864 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 35941 + reserve 4812 = 40753 MiB > 20480 MiB",
            "physical 0 + reserve 4812 = 4812 MiB <= 36864 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 16717 + reserve 4812 = 21529 MiB > 20480 MiB",
        ),
        (
            16000,
            60000,
            True,
            False,
            False,
            "physical 19224 + reserve 4812 = 24036 MiB > 16000 MiB; commit unchecked "
            "(source/workspace unmeasured); lower bound 35941 + reserve 4812 = 40753 MiB <= 60000 MiB",
            None,
        ),
        (
            36864,
            None,
            False,
            False,
            False,
            "physical 19224 + reserve 4812 = 24036 MiB <= 36864 MiB; commit unchecked (headroom unreadable)",
            None,
        ),
    ],
)
def test_measured_decisions_log_the_full_reason_and_sample_each_budget_once(
    tmp_path, monkeypatch, available, commit, lora, engaged, refuses, dense_check, route_check
):
    from core.inference import diffusion as dif

    records, calls = [], []

    class _LogSpy:
        def info(self, event, *args):
            records.append(event % args if args else event)

    def read(name, value):
        calls.append(name)
        return value

    monkeypatch.setattr(dm, "_host_ram_capacity_mib", lambda: read("total", 48128))
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: read("physical", available))
    monkeypatch.setattr(dm, "_windows_commit_headroom_mib", lambda: read("commit", commit))
    monkeypatch.setattr(dif, "stored_components", lambda snapshot: _qwen_companions())
    monkeypatch.setattr(dif, "file_size_mib", lambda path: 4096)
    monkeypatch.setattr(dif, "logger", _LogSpy())
    expected_reason = (
        "LoRA adapters need the dense denoiser; " if lora else ""
    ) + f"dense float16 load: {dense_check}"
    if route_check is not None:
        expected_reason += f"; low-memory route: {route_check}"
    backend = dif.DiffusionBackend()

    def decide():
        return backend._small_host_decision(
            str(tmp_path),
            {},
            types.SimpleNamespace(device = "cuda"),
            torch.float16,
            lora_active = lora,
            single_file_path = "diagnostic.gguf",
        )

    if refuses:
        with pytest.raises(RuntimeError, match = "Not enough system RAM") as error:
            decide()
        assert str(error.value) == (
            "Not enough system RAM to load this model: even the low-memory route exceeds a host memory budget: "
            f"{route_check}. Close other applications, pick a smaller or GGUF model, "
            f"or set {sh.HOST_RAM_CHECK_ENV}=0 to try anyway."
        )
    else:
        decision = decide()
        assert decision.engaged is engaged and decision.refuse is None
        assert decision.reason == expected_reason
    assert calls == ["total", "physical", "commit"]
    storage = {"text_encoder": "bfloat16"} if engaged else {}
    assert records == [
        f"diffusion.small_host: engaged={engaged} ({expected_reason}); stored-dtype components={storage}; "
        "prebuilt=4096 MiB (diagnostic only); source=16717 MiB; workspace dense=- MiB, route=- MiB"
    ]


def test_prebuilt_components_are_excluded_from_future_source_mappings(tmp_path, monkeypatch):
    from core.inference import diffusion as dif

    components = {
        "transformer": sh.StoredComponent("transformer", 38900, "bfloat16", source_mib = 38901),
        **_qwen_companions(),
    }
    monkeypatch.setattr(dif, "stored_components", lambda snapshot: components)
    monkeypatch.setattr(dif, "host_memory_budgets_mib", lambda: (48128, 36864, 60000))
    monkeypatch.setattr(dif, "file_size_mib", lambda path: 4096)
    decision = dif.DiffusionBackend()._small_host_decision(
        str(tmp_path),
        {"transformer": object()},
        types.SimpleNamespace(device = "cuda"),
        torch.float16,
        lora_active = False,
        single_file_path = "diagnostic.gguf",
    )
    assert not decision.engaged and decision.refuse is None
    assert decision.source_mapping_mib == 16717
    assert decision.dense_host_mib == 19224 and decision.route_host_mib == 0
    assert decision.prebuilt_host_mib == 4096
    # Production workspace is still unmeasured; diagnostics never masquerade as future copies.
    assert decision.dense_workspace_mib is None and decision.dense_commit_mib is None


@pytest.mark.parametrize("commit,refuses", [(30720, False), (20480, True)])
def test_real_loader_measures_after_transformer_and_refuses_before_companions(
    fake_runtime, tmp_path, monkeypatch, commit, refuses
):
    from core.inference import diffusion as dif
    from .test_diffusion_backend import _FakePipeline, _FakeTransformer

    calls = []

    class CompanionReached(Exception):
        pass

    def transformer(cls, *args, **kwargs):
        calls.append("transformer")
        return object()

    def measure():
        assert calls == ["transformer"]
        calls.append("measurement")
        return 48128, 36864, commit

    def companion(cls, *args, **kwargs):
        calls.append("companion")
        assert "transformer" in kwargs
        assert kwargs["torch_dtype"]["text_encoder"] is sys.modules["torch"].bfloat16
        raise CompanionReached("companion sentinel")

    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = dif.DiffusionBackend()
    monkeypatch.setattr(
        backend, "_pick_device_and_dtype", lambda: ("cuda", sys.modules["torch"].float16)
    )
    monkeypatch.setattr(_FakeTransformer, "from_single_file", classmethod(transformer))
    monkeypatch.setattr(_FakePipeline, "from_pretrained", classmethod(companion))
    monkeypatch.setattr(dif, "resolve_snapshot_dir", lambda *args: tmp_path)
    monkeypatch.setattr(dif, "stored_components", lambda snapshot: _qwen_companions())
    monkeypatch.setattr(dif, "host_memory_budgets_mib", measure)
    expected_error = RuntimeError if refuses else CompanionReached
    expected_text = "Not enough system RAM" if refuses else "companion sentinel"
    with pytest.raises(expected_error, match = expected_text):
        backend.load_pipeline(
            str(tmp_path),
            gguf_filename = "model.gguf",
            family_override = "qwen-image",
            base_repo = "base/repo",
            transformer_quant = "off",
            text_encoder_quant = "off",
            speed_mode = "eager",
            local_files_only = True,
        )
    assert calls == ["transformer", "measurement"] + ([] if refuses else ["companion"])


def test_lora_keeps_the_dense_load():
    d = _decide(FLUX1, lora_active = True)
    assert not d.engaged and d.refuse is None and "LoRA" in d.reason


def test_torch_dtype_map_only_names_routed_components():
    d = _decide(FLUX1)
    m = sh.torch_dtype_map(d, torch.float16)
    assert m == {
        "default": torch.float16,
        "transformer": torch.bfloat16,
        "text_encoder_2": torch.bfloat16,
    }
    assert sh.torch_dtype_map(_decide(FLUX1, available = 10**6), torch.float16) is torch.float16


def _write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, (dtype, nbytes) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": [nbytes // 2],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        fh.write(b"\0" * offset)


@pytest.mark.parametrize("stem", ["model", "diffusion_pytorch_model"])
def test_tensors_in_one_selected_view_count_the_full_file_once(tmp_path, stem):
    te = tmp_path / "text_encoder"
    te.mkdir()
    weight = te / f"{stem}.safetensors"
    _write_safetensors(weight, {"a": ("BF16", 1 << 19), "b": ("BF16", 1 << 19)})
    component = sh.stored_components(tmp_path)["text_encoder"]
    assert component.mib == 1
    # One MiB of tensors plus the header spans two MiB after rounding the whole mapped file.
    assert weight.stat().st_size > 1 << 20
    assert component.source_mib == 2
    assert sh._source_mapping_mib(te, [weight]) == 2


def test_shard_index_deduplicates_shared_views_but_distinct_loads_add(tmp_path):
    te = tmp_path / "text_encoder"
    te.mkdir()
    files = [te / "model-00001-of-00002.safetensors", te / "model-00002-of-00002.safetensors"]
    _write_safetensors(files[0], {"a": ("BF16", 128), "b": ("BF16", 128)})
    _write_safetensors(files[1], {"c": ("BF16", 128)})
    (te / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": files[0].name, "b": files[0].name, "c": files[1].name}}),
        encoding = "utf-8",
    )
    assert sh._source_mapping_mib(te, files) == 2
    # Distinct component loads get their own views even when selected source sizes are identical.
    components = {
        "text_encoder": sh.StoredComponent("text_encoder", 512, "bfloat16", source_mib = 2),
        "text_encoder_2": sh.StoredComponent("text_encoder_2", 512, "bfloat16", source_mib = 2),
    }
    decision = _commit_decide(components = components)
    assert decision.source_mapping_mib == 4


@pytest.mark.parametrize("layout", ["ambiguous", "missing", "traversal", "malformed", "unreadable"])
def test_unproved_source_selection_stays_unknown(tmp_path, layout):
    te = tmp_path / "text_encoder"
    te.mkdir()
    weight = te / "model.safetensors"
    _write_safetensors(weight, {"a": ("BF16", 128)})
    files = [weight]
    if layout == "ambiguous":
        other = te / "other.safetensors"
        _write_safetensors(other, {"b": ("BF16", 128)})
        files.append(other)
    elif layout == "unreadable":
        weight.write_bytes(b"not a safetensors header")
        assert "text_encoder" not in sh.stored_components(tmp_path)
        return
    else:
        weight.unlink()
        (te / "model.safetensors.index.json").write_text(
            "not JSON"
            if layout == "malformed"
            else json.dumps(
                {
                    "weight_map": {
                        "a": "../outside.safetensors" if layout == "traversal" else weight.name
                    }
                }
            ),
            encoding = "utf-8",
        )
    assert sh._source_mapping_mib(te, files) is None


def test_stored_components_read_headers_only(tmp_path):
    (tmp_path / "transformer").mkdir()
    (tmp_path / "vae").mkdir()
    (tmp_path / "scheduler").mkdir()
    _write_safetensors(tmp_path / "transformer" / "a.safetensors", {"w": ("BF16", 3 << 20)})
    _write_safetensors(tmp_path / "transformer" / "b.safetensors", {"x": ("F32", 1 << 20)})
    _write_safetensors(tmp_path / "vae" / "v.safetensors", {"w": ("F32", 2 << 20)})
    comps = sh.stored_components(tmp_path)
    assert comps["transformer"] == sh.StoredComponent("transformer", 4, "bfloat16")
    assert comps["vae"].dtype == "float32"
    assert "scheduler" not in comps
    assert sh.resolve_snapshot_dir(tmp_path) == tmp_path


def test_t5_kept_fp32_layers_are_budgeted_and_variants_ignored(tmp_path):
    te = tmp_path / "text_encoder_2"
    te.mkdir()
    (te / "config.json").write_text(json.dumps({"architectures": ["T5EncoderModel"]}))
    _write_safetensors(
        te / "model.safetensors",
        {
            "encoder.block.0.layer.1.DenseReluDense.wo.weight": ("BF16", 2 << 20),
            "encoder.block.0.layer.1.DenseReluDense.wi.weight": ("BF16", 6 << 20),
        },
    )
    _write_safetensors(te / "model.fp16.safetensors", {"w": ("F16", 8 << 20)})
    comps = sh.stored_components(tmp_path)
    assert comps["text_encoder_2"] == sh.StoredComponent("text_encoder_2", 8, "bfloat16", 2, 9)
    tiny = {
        "transformer": sh.StoredComponent("transformer", 4000, "bfloat16"),
        "text_encoder_2": sh.StoredComponent("text_encoder_2", 9084, "bfloat16", 1920),
    }
    # the streamed encoder's ``wo`` converts to fp32 on the host: 1920 MiB stored -> 3840 MiB loaded
    assert _decide(tiny).route_host_mib == 2000 + 3840


def test_int8_weight_storage_matches_the_dense_layer():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(2048, 4096), torch.nn.LayerNorm(4096), torch.nn.Linear(4096, 16)
    ).to(torch.bfloat16)
    ref = [m.weight.detach().float().clone() for m in (model[0], model[2])]
    stats = sh.quantize_int8_weight_(model, compute_dtype = torch.float32, work_device = "cpu")
    assert stats["linears"] == 1 and stats["dense_linears"] == 1
    big = model[0]
    assert isinstance(big, sh.int8_linear_class()) and big.qweight.dtype == torch.int8
    assert model[1].weight.dtype == torch.float32 and model[2].weight.dtype == torch.float32
    x = torch.randn(8, 2048)
    out = big(x)
    want = torch.nn.functional.linear(x, ref[0], model[0].bias.float())
    rel = (out - want).norm() / want.norm()
    assert rel < 0.01, rel


def test_streamed_encoder_keeps_linear_storage_and_reports_compute_dtype():
    pytest.importorskip("diffusers")
    from diffusers.hooks import HookRegistry

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wi = torch.nn.Linear(64, 128, bias = False)
            self.wo = torch.nn.Linear(128, 64, bias = False)

        def forward(self, x):
            # T5's pattern: the parent reads wo.weight.dtype BEFORE wo runs
            h = self.wi(x)
            return self.wo(h.to(self.wo.weight.dtype))

    class Enc(torch.nn.Module):
        _keep_in_fp32_modules = ["wo"]

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(100, 64)
            self.block = Block()
            self.norm = torch.nn.LayerNorm(64)
            # computed at init, never stored: fp32 in the dense fp16 load too (RoPE inv_freq)
            self.register_buffer("inv_freq", torch.rand(8), persistent = False)

        @property
        def dtype(self):
            return next(p.dtype for p in self.parameters() if p.is_floating_point())

        def forward(self, ids):
            return self.norm(self.block(self.embed(ids)))

    torch.manual_seed(0)
    enc = Enc()
    for p in enc.parameters():
        p.data = p.data.to(
            torch.bfloat16
        )  # the stored-dtype load: parameters bf16, init-time buffers fp32
    dense = Enc()
    dense.load_state_dict({k: v.float() for k, v in enc.state_dict().items()})
    dense.inv_freq = enc.inv_freq.clone()
    sh.prepare_streamed_encoder_(enc, torch.float16)
    assert enc.inv_freq.dtype == torch.float32
    enc_fp16 = enc
    enc = Enc()
    for p in enc.parameters():
        p.data = p.data.to(torch.bfloat16)
    dense.load_state_dict({k: v.float() for k, v in enc.state_dict().items()})
    dense.inv_freq = enc.inv_freq.clone()
    del enc_fp16
    sh.prepare_streamed_encoder_(enc, torch.float32)
    # the first float parameter's owner converts so .dtype reports the compute dtype
    assert enc.dtype == torch.float32
    assert enc.block.wi.weight.dtype == torch.bfloat16  # memory-mapped storage stays
    assert (
        HookRegistry.check_if_exists_or_initialize(enc.block.wi).get_hook("layerwise_casting")
        is not None
    )
    assert enc.block.wo.weight.dtype == torch.float32  # kept-fp32 converts now
    ids = torch.randint(0, 100, (2, 7))
    with torch.no_grad():
        assert torch.equal(enc(ids), dense(ids))  # bf16 -> fp32 is exact


def _group_plan(**kw):
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 14647, 15095)
    base = dict(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_MODEL,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = memory,
        estimates = {
            "safe_device_budget_mib": 12599,
            "runtime_headroom_mib": 3072,
            "base_overhead_mib": 2048,
        },
    )
    base.update(kw)
    return dm.MemoryPlan(**base)


def test_route_plan_streams_encoders_and_keeps_what_fits(monkeypatch):
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(
        dm,
        "_loaded_component_mib",
        lambda pipe: {
            "transformer": (11353, "dit"),
            "text_encoder_2": (9346, "text_encoder"),
            "vae": (160, "other"),
        },
    )
    pipe = types.SimpleNamespace(_unsloth_small_host = {"components": {}})
    log = types.SimpleNamespace(info = lambda *a, **k: None)
    plan = DiffusionBackend._small_host_plan(None, pipe, _group_plan(), None, log)
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_text_encoders
    assert plan.stream_transformer and plan.resident_transformer_mib == 12599 - 3072 - 2048 - 160
    # a plan that already keeps everything resident, or a pipe the route did not touch, is left alone
    none = _group_plan(offload_policy = dm.OFFLOAD_NONE)
    assert DiffusionBackend._small_host_plan(None, pipe, none, None, log) is none
    other = _group_plan()
    assert DiffusionBackend._small_host_plan(None, object(), other, None, log) is other


def test_lora_refused_and_status_reports_the_int8_route():
    from core.inference.diffusion import _small_host_int8

    pipe = types.SimpleNamespace(
        _unsloth_small_host = {"components": {"transformer": "int8 weights (1 MiB)"}}
    )
    assert _small_host_int8(pipe)
    assert not _small_host_int8(
        types.SimpleNamespace(_unsloth_small_host = {"components": {"transformer": "cast on device"}})
    )
    assert not _small_host_int8(object())
