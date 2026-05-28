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
BATCH_SIZE = 64
VOCAB_SIZE = 10000
ROPE_THETA = 10000.0
DTYPE = "float32"

WARMUP_STEPS = 5
PROFILING_STEPS = 10

# 如果你想严格满足题目 "context lengths larger than 128"，
# 建议把 ctx=128 的两组改成 ctx=1024，只要显存放得下。
CONFIGS = [
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 128},
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 256},
    {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12, "context_length": 512},

    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 128},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 256},
    {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "context_length": 512},
]


# =========================
# Paths
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK = PROJECT_ROOT / "benchmark.py"

OUT_DIR = PROJECT_ROOT / "nsys_profiles"
REPORT_DIR = OUT_DIR / "reports"
LOG_DIR = OUT_DIR / "logs"
SUMMARY_CSV = OUT_DIR / "profile_summary.csv"
SUMMARY_XLSX = OUT_DIR / "profile_summary.xlsx"
ANSWER_TEMPLATE = OUT_DIR / "answers_template.md"

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
        f"_bs{BATCH_SIZE}"
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
    report_file = REPORT_DIR / f"{run_id}.nsys-rep"
    sqlite_file = REPORT_DIR / f"{run_id}.sqlite"
    stdout_file = LOG_DIR / f"{run_id}.stdout.txt"
    stderr_file = LOG_DIR / f"{run_id}.stderr.txt"
    per_run_jsonl = LOG_DIR / f"{run_id}.profiling.jsonl"

    # Isolate benchmark.py's own timing log for this run.
    if BENCHMARK_LOG_FILE.exists():
        BENCHMARK_LOG_FILE.unlink()

    cmd = [
        "uv", "run",
        "nsys", "profile",
        "-o", str(report_base),
        "-f", "true",

        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",

        # all 的开销很大。一般看 kernel 来源和同步点，用 kernel,sync 就够。
        # 如果你确实想收集所有 CUDA API backtrace，可以改成 --cudabacktrace=all
        "--cudabacktrace=kernel,sync",
        "--python-backtrace=cuda",

        # 导出 sqlite 方便之后自动查表；同时 .nsys-rep 仍然可以用 GUI 打开。
        "--export=sqlite",
        "--stats=true",

        "--",
        "python", str(BENCHMARK),

        "--vocab_size", str(VOCAB_SIZE),
        "--batch_size", str(BATCH_SIZE),
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
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_ID

    print(f"\n========== Running {run_id} ==========")
    print(" ".join(cmd))

    with stdout_file.open("w", encoding="utf-8") as out, stderr_file.open("w", encoding="utf-8") as err:
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
        )

    if BENCHMARK_LOG_FILE.exists():
        shutil.copy2(BENCHMARK_LOG_FILE, per_run_jsonl)

    timing = parse_python_timing(BENCHMARK_LOG_FILE)

    row = {
        "run_id": run_id,
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,

        "d_model": cfg["d_model"],
        "d_ff": cfg["d_ff"],
        "num_layers": cfg["num_layers"],
        "num_heads": cfg["num_heads"],
        "context_length": cfg["context_length"],
        "batch_size": BATCH_SIZE,
        "vocab_size": VOCAB_SIZE,
        "dtype": DTYPE,
        "warmup_steps": WARMUP_STEPS,
        "profiling_steps": PROFILING_STEPS,

        "nsys_report": str(report_file),
        "nsys_sqlite": str(sqlite_file),
        "stdout_log": str(stdout_file),
        "stderr_log": str(stderr_file),
        "benchmark_jsonl": str(per_run_jsonl),

        # Python standard library timing from your benchmark.py
        "python_forward_mean_ms": timing.get("python_forward_mean_ms", pd.NA),
        "python_backward_mean_ms": timing.get("python_backward_mean_ms", pd.NA),
        "python_optimizer_step_mean_ms": timing.get("python_optimizer_step_mean_ms", pd.NA),
        "python_train_step_mean_ms": timing.get("python_train_step_mean_ms", pd.NA),
        "num_profiled_steps": timing.get("num_profiled_steps", pd.NA),

        # ========== Manual fields from Nsight Systems GUI ==========
        # Question (a)
        "nsys_forward_time_ms": pd.NA,
        "forward_matches_python_timing": pd.NA,
        "answer_a": pd.NA,

        # Question (b)
        "forward_top_cuda_kernel_by_cumulative_gpu_time": pd.NA,
        "forward_top_kernel_invocations_per_forward": pd.NA,
        "full_step_top_cuda_kernel_by_cumulative_gpu_time": pd.NA,
        "same_top_kernel_forward_vs_full_step": pd.NA,
        "answer_b": pd.NA,

        # Question (c)
        "non_matmul_forward_kernels_nontrivial": pd.NA,
        "answer_c": pd.NA,

        # Question (d)
        "forward_only_matmul_time_fraction_percent": pd.NA,
        "full_train_step_matmul_time_fraction_percent": pd.NA,
        "forward_only_other_kernel_fraction_percent": pd.NA,
        "full_train_step_other_kernel_fraction_percent": pd.NA,
        "answer_d": pd.NA,

        # Question (e)
        "attention_softmax_forward_time_ms": pd.NA,
        "attention_matmul_forward_time_ms": pd.NA,
        "attention_softmax_flops_estimate": pd.NA,
        "attention_matmul_flops_estimate": pd.NA,
        "softmax_vs_matmul_runtime_vs_flops_comment": pd.NA,
        "answer_e": pd.NA,
    }

    return row


def write_answer_template(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# Nsight Systems Profiling Answers Template\n")
    lines.append(
        "Fill in the Nsight Systems fields after opening each `.nsys-rep` file. "
        "For kernel questions, use **Stats System View → CUDA GPU Kernel Summary** and filter by NVTX ranges.\n"
    )

    for _, r in df.iterrows():
        lines.append(f"## {r['run_id']}\n")
        lines.append(f"- Report: `{r['nsys_report']}`")
        lines.append(f"- Config: d_model={r['d_model']}, d_ff={r['d_ff']}, "
                     f"layers={r['num_layers']}, heads={r['num_heads']}, "
                     f"context={r['context_length']}, batch={r['batch_size']}")
        lines.append(f"- Python forward mean: `{r['python_forward_mean_ms']}` ms")
        lines.append(f"- Python backward mean: `{r['python_backward_mean_ms']}` ms")
        lines.append(f"- Python optimizer step mean: `{r['python_optimizer_step_mean_ms']}` ms\n")

        lines.append("**(a)** Forward pass time and comparison with Python timing:\n")
        lines.append("> TODO\n")

        lines.append("**(b)** Top cumulative CUDA kernel in forward pass, invocation count, and comparison with full step:\n")
        lines.append("> TODO\n")

        lines.append("**(c)** Non-matmul kernels with non-trivial forward runtime:\n")
        lines.append("> TODO\n")

        lines.append("**(d)** Matrix multiplication fraction in forward-only vs complete training step:\n")
        lines.append("> TODO\n")

        lines.append("**(e)** Softmax vs matmul runtime inside self-attention, compared with FLOPs:\n")
        lines.append("> TODO\n")

    ANSWER_TEMPLATE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = []

    for cfg in CONFIGS:
        row = run_one_profile(cfg)
        rows.append(row)

        # Incrementally save after every run, so you keep partial results if a later run OOMs.
        df = pd.DataFrame(rows)
        df.to_csv(SUMMARY_CSV, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(SUMMARY_CSV, index=False)

    try:
        df.to_excel(SUMMARY_XLSX, index=False)
    except Exception as e:
        print(f"Could not write xlsx file: {e}")

    write_answer_template(df)

    print("\n========== Done ==========")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary XLSX: {SUMMARY_XLSX}")
    print(f"Answer template: {ANSWER_TEMPLATE}")
    print(f"Reports: {REPORT_DIR}")


if __name__ == "__main__":
    main()