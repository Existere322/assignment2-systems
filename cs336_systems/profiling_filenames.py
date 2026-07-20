from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass


def _format_value(value: object) -> str:
    """Convert one option value into a filesystem-safe name component."""
    token = re.sub(r"[^A-Za-z0-9.-]+", "-", str(value)).strip("-")
    if not token:
        raise ValueError(f"Cannot use {value!r} in a profiling filename")
    return token


def _format_flag(value: object) -> str:
    return "1" if int(value) else "0"


def _format_precision(value: object) -> str:
    return "bf16" if int(value) else "fp32"


@dataclass(frozen=True)
class FilenameField:
    option: str
    prefix: str = ""
    formatter: Callable[[object], str] = _format_value


# This is the single place that controls which benchmark/profiling options are
# encoded in every output filename. Add a FilenameField here when a new option
# must distinguish profiling runs; run_nsys.py and benchmark.py both use it.
PROFILE_FILENAME_FIELDS = (
    # FilenameField("new_option", "new-prefix")
    FilenameField("d_model", "d"),
    FilenameField("d_ff", "ff"),
    FilenameField("num_layers", "L"),
    FilenameField("num_heads", "H"),
    FilenameField("context_length", "ctx"),
    FilenameField("batch_size", "bs"),
    FilenameField("dtype", "dtype"),
    FilenameField("warmup_steps", "warmup"),
    FilenameField("profiling_steps", "steps"),
    FilenameField("profiling_warmup", "profile-warmup", _format_flag),
    FilenameField("profile_attn", "attn", _format_flag),
    FilenameField("use_mixed_precision", formatter=_format_precision),
    FilenameField("use_memory_profiling", "memory", _format_flag),
    FilenameField("run_mode"),
    FilenameField("use_checkpoints", "checkpoints", _format_flag),
    FilenameField("per_checkpoint_layers", "checkpoint-layers"),
    FilenameField("use_jit_compiler", "compiled", _format_flag), 
)


def make_profile_name(options: Mapping[str, object]) -> str:
    """Build the canonical name shared by all outputs for one profiling run."""
    missing = [field.option for field in PROFILE_FILENAME_FIELDS if field.option not in options]
    if missing:
        missing_options = ", ".join(f"--{option}" for option in missing)
        raise ValueError(f"Missing profiling filename options: {missing_options}")

    return "_".join(
        f"{field.prefix}{field.formatter(options[field.option])}"
        for field in PROFILE_FILENAME_FIELDS
    )
