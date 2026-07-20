from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from statistics import mean


from cs336_systems.profiling_filenames import make_profile_name


# =========================
# Basic settings
# =========================

GPU_ID = "0"
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
DTYPE = "float32"
WARMUP_STEPS = 5
PROFILING_STEPS = 15
PROFILING_WARMUP = False
PROFILE_ATTN = True
MIXED_PRECISION = False
MEMORY_PROFILING = False
INFERENCE_ONLY = False
USE_JIT_COMPILER = False
USE_CHECKPOINTS = False
PER_CHECKPOINT_LAYERS = 1

# CPU/CUDA backtraces require both perf_event_open permission and access to a
# symbol server. Keep them disabled by default because this machine currently
# has perf_event_paranoid=4 and cannot reach debuginfod.ubuntu.com.
ENABLE_BACKTRACES = False

CONFIGS = [
    # {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 128, "batch_size": 4},
    # {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 256, "batch_size": 4},
    # {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512, "batch_size": 4},
    # {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 1024, "batch_size": 4}, # 1,391,569,403,904
    # {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 128, "batch_size": 4},
    # {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 256, "batch_size": 4},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512, "batch_size": 4}, # 3,298,534,883,328
    # {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 1024, "batch_size": 4}, 
    # {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32, "context_length": 32, "batch_size": 4}    # 3,435,973,836,800
]


# =========================
# Paths
# =========================

mixed_precision_re = "bf16" if MIXED_PRECISION else "fp32"
run_mode = "inference" if INFERENCE_ONLY else "train"
PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK = PROJECT_ROOT / "benchmark.py"

OUT_DIR = PROJECT_ROOT / "nsys_profiles"
REPORT_DIR = OUT_DIR / f"reports_{mixed_precision_re}_{run_mode}"
LOG_DIR = OUT_DIR / "logs"

BENCHMARK_LOG_DIR = PROJECT_ROOT / "profiling_results"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARK_LOG_DIR.mkdir(parents=True, exist_ok=True)


def make_benchmark_args(cfg: dict) -> dict[str, object]:
    """Return the complete benchmark CLI configuration for one run."""
    return {
        "vocab_size": VOCAB_SIZE,
        "batch_size": cfg["batch_size"],
        "context_length": cfg["context_length"],
        "d_model": cfg["d_model"],
        "d_ff": cfg["d_ff"],
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
        "rope_theta": ROPE_THETA,
        "dtype": DTYPE,
        "warmup_steps": WARMUP_STEPS,
        "profiling_steps": PROFILING_STEPS,
        "profiling_warmup": int(PROFILING_WARMUP),
        "profile_attn": int(PROFILE_ATTN),
        "use_mixed_precision": int(MIXED_PRECISION),
        "use_memory_profiling": int(MEMORY_PROFILING),
        "run_mode": run_mode,
        "use_checkpoints": int(USE_CHECKPOINTS),
        "per_checkpoint_layers": PER_CHECKPOINT_LAYERS,
        "use_jit_compiler": int(USE_JIT_COMPILER)
    }


def make_cli_args(options: dict[str, object]) -> list[str]:
    """Convert benchmark option names and values to command-line arguments."""
    return [item for name, value in options.items() for item in (f"--{name}", str(value))]


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


def run_one_profile(cfg: dict) -> None:
    benchmark_args = make_benchmark_args(cfg)
    run_id = make_profile_name(benchmark_args)
    report_base = REPORT_DIR / run_id
    # stdout_file = LOG_DIR / f"{run_id}.stdout.txt"
    # Isolate benchmark.py's own timing log for this run.
    benchmark_log_file = BENCHMARK_LOG_DIR / f"profiling_{run_id}.jsonl"
    stderr_file = LOG_DIR / f"{run_id}.stderr.txt"
    per_run_jsonl = LOG_DIR / f"{run_id}.profiling.jsonl"
    if benchmark_log_file.exists():
        benchmark_log_file.unlink()

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
        # "--pytorch=functions-trace,autograd-shapes-nvtx",
        "--cuda-memory-usage=true", 

        *profiling_options,

        "--stats=true",

        "--",
        "python", str(BENCHMARK),
        *make_cli_args(benchmark_args),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID
    # env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # 其他实验可以关闭
    if not ENABLE_BACKTRACES:
        # DEBUGINFOD_URLS enables remote debug-symbol downloads. Removing it is
        # a second guard against blocking in debuginfod when symbols are off.
        env.pop("DEBUGINFOD_URLS", None)

    print(f"\n========== Running {run_id} ==========")
    print(" ".join(cmd))
    with stderr_file.open("w", encoding="utf-8") as err:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stderr=err,
            text=True,
        )

    if benchmark_log_file.exists():
        shutil.copy2(benchmark_log_file, per_run_jsonl)


def main() -> None:

    for cfg in CONFIGS:
        run_one_profile(cfg)

    print("\n========== Done ==========")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()