# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform model-weight load progress for a spawned sd-server process.

The image engine starts sd-server with the diffusion GGUF, its LLM text encoder,
the VAE and the tokenizer. Reading those files is the long first-generate phase,
and sd-server reports no progress for it. Sample the process read bytes plus its
resident set against the sum of the model files instead: the read counter covers
weights streamed straight to VRAM, the resident set covers mmap page-in, and the
sampler keeps the larger value, so pages released after the upload never pull the
bar back.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Optional

MODEL_FLAGS = ("--diffusion-model", "--llm", "--vae", "--tokenizer", "-m", "--model")
_VRAM_CACHE_SECONDS = 2.0
# The read counter can stop short of the file sum (mmap'd parts, files the server never reads
# fully). After this long without growth the load counts as finished; the caller also hides the
# bar on the first denoising step.
_IDLE_READY_SECONDS = 15.0
# Growth below this many bytes between two samples is noise, not progress.
_GROWTH_BYTES = 100_000


def model_file_bytes(args: Optional[Iterable[str]]) -> int:
    """Sum of the model files named on the server command line, 0 when none exist."""
    if not args:
        return 0
    tokens = [str(arg) for arg in args]
    paths: list[str] = []
    for flag in MODEL_FLAGS:
        for index, token in enumerate(tokens[:-1]):
            if token == flag:
                paths.append(tokens[index + 1])
    for token in tokens:
        if token.lower().endswith(".gguf"):
            paths.append(token)
    total = 0
    seen: set[str] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        try:
            if os.path.isfile(path):
                total += os.path.getsize(path)
        except OSError:
            continue
    return total


if sys.platform == "win32":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error = True)
    _psapi = ctypes.WinDLL("psapi", use_last_error = True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.GetProcessIoCounters.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_IoCounters),
    ]
    _kernel32.GetProcessIoCounters.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    _psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

    def _open_process(pid: int) -> wintypes.HANDLE:
        return _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)


def _darwin_usage(pid: int) -> Optional[tuple[int, int]]:
    """``(diskio_bytesread, phys_footprint)`` of ``pid`` via the existing libproc helper."""
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend._darwin_rusage(pid)


def process_read_bytes(pid: int) -> Optional[int]:
    """Bytes ``pid`` has taken in; None where the OS cannot report it.

    macOS uses ``ri_diskio_bytesread`` from the existing libproc helper.
    """
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/io", "r", encoding = "utf-8") as handle:
                for line in handle:
                    if line.startswith("rchar:"):
                        return int(line.split()[1])
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return None
    if sys.platform == "darwin":
        usage = _darwin_usage(pid)
        return None if usage is None else usage[0]
    if sys.platform != "win32":
        return None
    handle = _open_process(pid)
    if not handle:
        return None
    try:
        counters = _IoCounters()
        if not _kernel32.GetProcessIoCounters(handle, ctypes.byref(counters)):
            return None
        return int(counters.ReadTransferCount)
    finally:
        _kernel32.CloseHandle(handle)


def process_rss_bytes(pid: int) -> Optional[int]:
    """Resident bytes of ``pid``.

    File-backed on Linux, physical footprint on macOS; None where unavailable.
    """
    if os.path.isdir("/proc"):
        rss = None
        try:
            with open(f"/proc/{pid}/status", "r", encoding = "utf-8") as handle:
                for line in handle:
                    if line.startswith("RssFile:"):
                        return int(line.split()[1]) * 1024
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) * 1024
        except (FileNotFoundError, PermissionError, ValueError, IndexError, OSError):
            return None
        return rss
    if sys.platform == "darwin":
        usage = _darwin_usage(pid)
        return None if usage is None else usage[1]
    if sys.platform != "win32":
        return None
    handle = _open_process(pid)
    if not handle:
        return None
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        if not _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    finally:
        _kernel32.CloseHandle(handle)


class LoadProgressSampler:
    """Per-pid high-water mark so a dropped RSS (after the VRAM upload) never regresses.

    The VRAM query runs on a daemon thread and only its cached value reaches
    ``snapshot()``: the generate-progress poll runs on the backend event loop,
    where a synchronous nvidia-smi call would stall every other request.
    """

    def __init__(self) -> None:
        self._high_water: dict[int, int] = {}
        self._prev_loaded: dict[int, int] = {}
        self._last_growth: dict[int, float] = {}
        self._pid: Optional[int] = None
        self._vram_lock = threading.Lock()
        self._vram_pid: Optional[int] = None
        self._vram_value: Optional[int] = None
        self._vram_at = 0.0
        self._vram_refreshing = False

    def snapshot(
        self,
        pid: int,
        args: Optional[Iterable[str]],
        *,
        total_bytes: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        total = model_file_bytes(args) if total_bytes is None else total_bytes
        self._bind_pid(pid)
        read = process_read_bytes(pid)
        rss = process_rss_bytes(pid)
        if read is None and rss is None:
            return None
        now = time.monotonic()
        loaded = max(read or 0, rss or 0)
        loaded = max(loaded, self._high_water.get(pid, 0))
        self._high_water[pid] = loaded
        previous = self._prev_loaded.get(pid)
        if previous is None or loaded > previous + _GROWTH_BYTES:
            # The first sample starts the clock, so a load whose counters never
            # grow still becomes ready after the window instead of hanging below 100%.
            self._last_growth[pid] = now
        self._prev_loaded[pid] = loaded
        if total > 0:
            # Reads past the weights (headers, a second pass) must not pass the total.
            loaded = min(loaded, total)
        fraction = min(1.0, loaded / total) if total > 0 else 0.0
        idle = now - self._last_growth.get(pid, now)
        ready = fraction >= 1.0 or idle >= _IDLE_READY_SECONDS
        return {
            "phase": "ready" if ready else "loading",
            "bytes_loaded": loaded,
            "bytes_total": total,
            "fraction": round(fraction, 4),
            "vram_mb": self._vram_cached(pid),
            "eta_seconds": None,
        }

    def _bind_pid(self, pid: int) -> None:
        """Drop marks of a previous server: a reused pid must not inherit its progress."""
        if pid == self._pid:
            return
        self._pid = pid
        self._high_water.clear()
        self._prev_loaded.clear()
        self._last_growth.clear()
        with self._vram_lock:
            self._vram_pid = pid
            self._vram_value = None
            self._vram_at = 0.0

    def _vram_cached(self, pid: int) -> Optional[int]:
        """Last sampled VRAM of ``pid``; a stale cache schedules a background refresh."""
        now = time.monotonic()
        with self._vram_lock:
            if self._vram_pid != pid:
                return None
            value = self._vram_value
            fresh = now - self._vram_at < _VRAM_CACHE_SECONDS
            spawn = not fresh and not self._vram_refreshing
            if spawn:
                self._vram_refreshing = True
        if spawn:
            threading.Thread(
                target = self._refresh_vram,
                args = (pid,),
                name = "sd-load-vram",
                daemon = True,
            ).start()
        return value

    def _refresh_vram(self, pid: int) -> None:
        try:
            value = self._query_vram(pid)
        except Exception:
            value = None
        with self._vram_lock:
            if pid == self._vram_pid:
                self._vram_value = value
                self._vram_at = time.monotonic()
            self._vram_refreshing = False

    def _query_vram(self, pid: int) -> Optional[int]:
        """VRAM of ``pid`` from nvidia-smi; never raises (a missing tool is just None)."""
        value: Optional[int] = None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output = True,
                text = True,
                timeout = 3,
            )
            for line in (out.stdout or "").splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                    value = int(parts[1])
                    break
        except Exception:
            value = None
        return value
