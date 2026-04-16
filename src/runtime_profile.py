"""Hardware-aware runtime profiling helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import sys
from typing import Literal


DeviceType = Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True)
class RuntimeProfile:
    """Runtime settings tuned to the available hardware."""

    device: DeviceType
    device_reason: str
    cpu_threads: int
    numba_threads: int
    embedding_batch_size: int
    chunk_size: int
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    system_gpu_name: str | None = None
    system_gpu_memory_gb: float | None = None


def _default_cpu_threads(logical_cores: int) -> int:
    return max(1, logical_cores)


def _default_chunk_size(logical_cores: int) -> int:
    if logical_cores >= 24:
        return 350_000
    if logical_cores >= 16:
        return 300_000
    if logical_cores >= 8:
        return 180_000
    return 80_000


def _default_embedding_batch_size(
    device: DeviceType,
    gpu_memory_gb: float | None,
    cpu_threads: int,
) -> int:
    if device == "cuda":
        if gpu_memory_gb is None:
            return 160
        if gpu_memory_gb >= 20:
            return 640
        if gpu_memory_gb >= 12:
            return 384
        if gpu_memory_gb >= 8:
            return 224
        if gpu_memory_gb >= 6:
            return 128
        return 96

    if device == "mps":
        return 128

    if cpu_threads >= 16:
        return 64
    if cpu_threads >= 8:
        return 32
    return 16


def _probe_nvidia_smi() -> tuple[str | None, float | None]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None, None

    try:
        completed = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except Exception:
        return None, None

    first_line = next(
        (line.strip() for line in completed.stdout.splitlines() if line.strip()),
        "",
    )
    if not first_line:
        return None, None

    parts = [part.strip() for part in first_line.split(",", maxsplit=1)]
    name = parts[0] if parts else None
    memory_gb = None
    if len(parts) > 1:
        try:
            memory_gb = round(float(parts[1]) / 1024.0, 2)
        except ValueError:
            memory_gb = None
    return name, memory_gb


def detect_runtime_profile(
    device: str = "auto",
    cpu_threads: int | None = None,
    embed_batch_size: int | None = None,
    chunk_size: int | None = None,
) -> RuntimeProfile:
    """Detect an adaptive runtime profile for the current machine."""

    logical_cores = max(os.cpu_count() or 1, 1)
    chosen_cpu_threads = cpu_threads or _default_cpu_threads(logical_cores)
    chosen_cpu_threads = max(1, min(chosen_cpu_threads, logical_cores))

    torch_module = None
    torch_cuda_available = False
    torch_mps_available = False
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    system_gpu_name, system_gpu_memory_gb = _probe_nvidia_smi()

    try:
        import torch  # type: ignore[import-not-found]

        torch_module = torch
        torch_cuda_available = bool(torch.cuda.is_available())
        torch_mps_available = bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        if torch_cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
                2,
            )
        elif torch_mps_available:
            gpu_name = "Apple Silicon GPU"
    except Exception:
        torch_module = None

    requested_device = device.lower().strip()
    chosen_device: DeviceType = "cpu"
    device_reason = "CPU execution selected."

    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        requested_device = "auto"

    if requested_device == "cuda":
        if torch_cuda_available:
            chosen_device = "cuda"
            device_reason = "Using CUDA for SentenceTransformer inference."
        else:
            chosen_device = "cpu"
            if system_gpu_name:
                device_reason = (
                    f"CUDA requested but torch cannot use the detected GPU ({system_gpu_name}); "
                    "falling back to CPU."
                )
            else:
                device_reason = "CUDA requested but not available in the current torch runtime; falling back to CPU."
    elif requested_device == "mps":
        if torch_mps_available:
            chosen_device = "mps"
            device_reason = "Using MPS for SentenceTransformer inference."
        else:
            chosen_device = "cpu"
            device_reason = "MPS requested but not available; falling back to CPU."
    elif requested_device == "cpu":
        chosen_device = "cpu"
        device_reason = "CPU execution selected by override."
    else:
        if torch_cuda_available:
            chosen_device = "cuda"
            device_reason = "Auto-detected CUDA for SentenceTransformer inference."
        elif torch_mps_available:
            chosen_device = "mps"
            device_reason = "Auto-detected MPS for SentenceTransformer inference."
        else:
            chosen_device = "cpu"
            if system_gpu_name:
                device_reason = (
                    f"Detected GPU ({system_gpu_name}) but torch cannot use CUDA here; "
                    "falling back to CPU."
                )
            else:
                device_reason = "No supported GPU backend available; using CPU."

    effective_gpu_name = gpu_name if chosen_device != "cpu" else None
    effective_gpu_memory = gpu_memory_gb if chosen_device == "cuda" else None
    chosen_embed_batch = embed_batch_size or _default_embedding_batch_size(
        chosen_device,
        effective_gpu_memory,
        chosen_cpu_threads,
    )
    chosen_chunk_size = chunk_size or _default_chunk_size(logical_cores)

    # Touch torch once more so CPU-only runs scale to available cores.
    if torch_module is not None:
        try:
            torch_module.set_num_threads(chosen_cpu_threads)
        except Exception:
            pass
        try:
            interop_threads = max(1, min(4, chosen_cpu_threads // 2))
            torch_module.set_num_interop_threads(interop_threads)
        except Exception:
            pass

    return RuntimeProfile(
        device=chosen_device,
        device_reason=device_reason,
        cpu_threads=chosen_cpu_threads,
        numba_threads=chosen_cpu_threads,
        embedding_batch_size=max(1, chosen_embed_batch),
        chunk_size=max(10_000, chosen_chunk_size),
        gpu_name=effective_gpu_name,
        gpu_memory_gb=effective_gpu_memory,
        system_gpu_name=system_gpu_name,
        system_gpu_memory_gb=system_gpu_memory_gb,
    )


def apply_runtime_profile(profile: RuntimeProfile) -> None:
    """Apply threading-related settings for CPU-bound libraries."""

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(profile.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(profile.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(profile.cpu_threads)

    # Numba is often imported transitively by UMAP before we have a chance to
    # build the runtime profile. Changing NUMBA_NUM_THREADS after import can
    # trigger a runtime error once the thread pool has been initialized.
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(profile.numba_threads)

    try:
        import torch  # type: ignore[import-not-found]

        try:
            torch.set_num_threads(profile.cpu_threads)
        except Exception:
            pass
        try:
            interop_threads = max(1, min(8, profile.cpu_threads))
            torch.set_num_interop_threads(interop_threads)
        except Exception:
            pass
        if profile.device == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
    except Exception:
        pass

    try:
        import numba  # type: ignore[import-not-found]

        numba.set_num_threads(profile.numba_threads)
    except Exception:
        pass


def format_runtime_profile(profile: RuntimeProfile) -> str:
    """Return a user-friendly summary of the selected runtime settings."""

    gpu_bits: list[str] = []
    if profile.gpu_name:
        gpu_bits.append(profile.gpu_name)
    if profile.gpu_memory_gb is not None:
        gpu_bits.append(f"{profile.gpu_memory_gb:.2f} GB")
    gpu_summary = f" | GPU: {', '.join(gpu_bits)}" if gpu_bits else ""
    return (
        f"Runtime profile -> device={profile.device}, cpu_threads={profile.cpu_threads}, "
        f"numba_threads={profile.numba_threads}, embed_batch_size={profile.embedding_batch_size}, "
        f"chunk_size={profile.chunk_size}{gpu_summary}\n{profile.device_reason}"
    )
