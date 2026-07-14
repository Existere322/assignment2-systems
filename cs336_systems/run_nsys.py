from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean

import pandas as pd


# =========================
# Basic settings
# =========================

GPU_ID = "0"
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
DTYPE = "float32"
WARMUP_STEPS = 5
PROFILING_STEPS = 15
MIXED_PRECISION = True
MEMORY_PROFILING = False

# CPU/CUDA backtraces require both perf_event_open permission and access to a
# symbol server. Keep them disabled by default because this machine currently
# has perf_event_paranoid=4 and cannot reach debuginfod.ubuntu.com.
ENABLE_BACKTRACES = False

CONFIGS = [
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 128, "batch_size": 4},
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 256, "batch_size": 4},
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512, "batch_size": 4},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 128, "batch_size": 4},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 256, "batch_size": 4},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512, "batch_size": 4},
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 2048, "batch_size": 4},
]


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK = PROJECT_ROOT / "benchmark.py"

OUT_DIR = PROJECT_ROOT / "nsys_profiles"
REPORT_DIR = OUT_DIR / "reports"
if MIXED_PRECISION: REPORT_DIR = OUT_DIR / "reports_mixed_precision"
LOG_DIR = OUT_DIR / "logs"

BENCHMARK_LOG_DIR = PROJECT_ROOT / "profiling_results"
BENCHMARK_LOG_FILE = BENCHMARK_LOG_DIR / "profiling.jsonl"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_run_id(cfg: dict) -> str:
    return (
        f"d{cfg['d_model']}"
        f"_ff{cfg['d_ff']}"
        f"_L{cfg['num_layers']}"
        f"_H{cfg['num_heads']}"
        f"_ctx{cfg['context_length']}"
        f"_bs{cfg['batch_size']}"
    )


def parse_python_timing(log_file: Path) -> dict:
    """
    Parse benchmark.py's profiling_results/profiling.jsonl.

    Expected step rows:
    {"step": ..., "forward_time": ..., "backward_time": ..., "step_time": ...}
    Times are in seconds.
    """
    if not log_file.exists():
        return {}

    step_rows = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if {"forward_time", "backward_time", "step_time"} <= obj.keys():
                step_rows.append(obj)

    if not step_rows:
        return {}

    forward_ms = [r["forward_time"] * 1000 for r in step_rows]
    backward_ms = [r["backward_time"] * 1000 for r in step_rows]
    opt_step_ms = [r["step_time"] * 1000 for r in step_rows]

    return {
        "python_forward_mean_ms": mean(forward_ms),
        "python_backward_mean_ms": mean(backward_ms),
        "python_optimizer_step_mean_ms": mean(opt_step_ms),
        "python_train_step_mean_ms": mean(
            f + b + s for f, b, s in zip(forward_ms, backward_ms, opt_step_ms)
        ),
        "num_profiled_steps": len(step_rows),
    }


def run_one_profile(cfg: dict) -> dict:
    run_id = make_run_id(cfg)
    report_base = REPORT_DIR / run_id
    # stdout_file = LOG_DIR / f"{run_id}.stdout.txt"
    stderr_file = LOG_DIR / f"{run_id}.stderr.txt"
    per_run_jsonl = LOG_DIR / f"{run_id}.profiling.jsonl"

    if MIXED_PRECISION:
        stderr_file = LOG_DIR / f"{run_id}.stderr_mixed_precision.txt"
        per_run_jsonl = LOG_DIR / f"{run_id}.profiling_mixed_precision.jsonl"

    # Isolate benchmark.py's own timing log for this run.
    if BENCHMARK_LOG_FILE.exists():
        BENCHMARK_LOG_FILE.unlink()

    if ENABLE_BACKTRACES:
        profiling_options = [
            "--sample=process-tree",
            "--cpuctxsw=process-tree",
            "--resolve-symbols=true",
            "--cudabacktrace=kernel,sync",
            "--python-backtrace=cuda",
        ]
    else:
        profiling_options = [
            "--sample=none",
            "--cpuctxsw=none",
            "--resolve-symbols=false",
        ]

    cmd = [
        "uv", "run",
        "nsys", "profile",
        "-o", str(report_base),
        "-f", "true",

        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",

        *profiling_options,

        "--stats=true",

        "--",
        "python", str(BENCHMARK),

        "--vocab_size", str(VOCAB_SIZE),
        "--batch_size", str(cfg["batch_size"]),
        "--context_length", str(cfg["context_length"]),
        "--d_model", str(cfg["d_model"]),
        "--d_ff", str(cfg["d_ff"]),
        "--num_layers", str(cfg["num_layers"]),
        "--num_heads", str(cfg["num_heads"]),
        "--rope_theta", str(ROPE_THETA),
        "--dtype", DTYPE,

        "--warmup_steps", str(WARMUP_STEPS),
        "--profiling_steps", str(PROFILING_STEPS),

        # 让模型内部 attention 相关 profiling / NVTX 标记打开，前提是你的 BasicsTransformerLM 使用了它。
        "--profile_attn", "1",
        "--use_mixed_precision", str(1 if MIXED_PRECISION else 0), 
        "--use_memory_profiling", str(1 if MEMORY_PROFILING else 0), 
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    if not ENABLE_BACKTRACES:
        # DEBUGINFOD_URLS enables remote debug-symbol downloads. Removing it is
        # a second guard against blocking in debuginfod when symbols are off.
        env.pop("DEBUGINFOD_URLS", None)

    print(f"\n========== Running {run_id} ==========")
    print(" ".join(cmd))
    with stderr_file.open("w", encoding="utf-8") as err:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stderr=err,
            text=True,
        )

    if BENCHMARK_LOG_FILE.exists():
        shutil.copy2(BENCHMARK_LOG_FILE, per_run_jsonl)


def main() -> None:

    for cfg in CONFIGS:
        run_one_profile(cfg)

    print("\n========== Done ==========")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()