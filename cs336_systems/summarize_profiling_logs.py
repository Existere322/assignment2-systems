"""Summarize fp32 and mixed-precision profiling JSONL logs with pandas.

By default, the script reads ``nsys_profiles/logs/**/*.jsonl``, writes the
table to ``nsys_profiles/profiling_summary.csv``, and draws the total average
time comparison in ``nsys_profiles/profiling_total_time.png``.  Timing values
remain in seconds, which is the unit used by the profiling logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "nsys_profiles" / "logs"
DEFAULT_OUTPUT = PROJECT_ROOT / "nsys_profiles" / "profiling_summary.csv"
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "nsys_profiles" / "profiling_total_time.png"

CONFIG_COLUMNS = [
    "d_model",
    "d_ff",
    "context_length",
    "batch_size",
    "num_layers",
    "num_heads",
]
TIMING_COLUMNS = ["forward_time", "backward_time", "optimizer_time"]
MODEL_SPEC_COLUMN = (
    "模型规格(d_model,d_ff,context_len,batch_size,num_layers,num_heads)"
)
OUTPUT_COLUMNS = [
    MODEL_SPEC_COLUMN,
    "precision",
    "avg_forward_time",
    "avg_backward_time",
    "avg_optimizer_time",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate nsys profiling JSONL files into a CSV table."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory searched recursively for JSONL files (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DEFAULT_PLOT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_PLOT_OUTPUT})",
    )
    parser.add_argument(
        "--time-unit",
        choices=("s", "ms"),
        default="ms",
        help="Output timing unit: seconds or milliseconds (default: s)",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=6,
        help="Number of decimal places in the output (default: 6)",
    )
    return parser.parse_args()


def precision_name(config: pd.Series, log_path: Path) -> str:
    """Return a readable precision label from config, with a filename fallback."""
    mixed_value = config.get("use_mixed_precision")
    if pd.notna(mixed_value):
        try:
            return "mixed precision (bf16)" if int(mixed_value) == 1 else "fp32"
        except (TypeError, ValueError):
            value = str(mixed_value).strip().lower()
            if value in {"true", "yes", "mixed", "mixed_precision"}:
                return "mixed precision (bf16)"
            if value in {"false", "no", "fp32", "float32"}:
                return "fp32"

    dtype = str(config.get("dtype", "")).strip().lower()
    if dtype in {"float16", "fp16", "bfloat16", "bf16"}:
        return dtype
    if "mixed_precision" in log_path.stem:
        return "mixed precision (bf16)"
    return "fp32"


def summarize_log(log_path: Path, time_scale: float) -> dict[str, object]:
    """Summarize one JSONL file into one output-table row."""
    frame = pd.read_json(log_path, lines=True)

    missing_config = [column for column in CONFIG_COLUMNS if column not in frame]
    if missing_config:
        raise ValueError(f"missing config fields: {', '.join(missing_config)}")

    missing_timings = [column for column in TIMING_COLUMNS if column not in frame]
    if missing_timings:
        raise ValueError(f"missing timing fields: {', '.join(missing_timings)}")

    config_rows = frame.dropna(subset=CONFIG_COLUMNS)
    if config_rows.empty:
        raise ValueError("no model configuration row")
    config = config_rows.iloc[0]

    timings = frame[TIMING_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
    if timings.empty:
        raise ValueError("no complete timing rows")

    model_values = tuple(int(config[column]) for column in CONFIG_COLUMNS)
    averages = timings.mean() * time_scale

    return {
        MODEL_SPEC_COLUMN: str(model_values),
        "precision": precision_name(config, log_path),
        "avg_forward_time": averages["forward_time"],
        "avg_backward_time": averages["backward_time"],
        "avg_optimizer_time": averages["optimizer_time"],
        "_sort_key": (*model_values, precision_name(config, log_path)),
    }


def build_summary(log_dir: Path, time_unit: str) -> pd.DataFrame:
    """Read all JSONL logs recursively and return the requested summary table."""
    if not log_dir.is_dir():
        raise FileNotFoundError(f"log directory does not exist: {log_dir}")

    time_scale = 1000.0 if time_unit == "ms" else 1.0
    records: list[dict[str, object]] = []

    for log_path in sorted(log_dir.rglob("*.jsonl")):
        try:
            records.append(summarize_log(log_path, time_scale))
        except (OSError, ValueError, TypeError) as exc:
            print(f"warning: skipped {log_path}: {exc}", file=sys.stderr)

    if not records:
        raise ValueError(f"no usable JSONL profiling logs found under {log_dir}")

    records.sort(key=lambda record: record["_sort_key"])
    return pd.DataFrame(records)[OUTPUT_COLUMNS]


def plot_total_average_time(
    summary: pd.DataFrame, output_path: Path, time_unit: str
) -> None:
    """Draw grouped bars comparing total average time for both precisions."""
    plot_data = summary.assign(
        total_average_time=summary[
            ["avg_forward_time", "avg_backward_time", "avg_optimizer_time"]
        ].sum(axis=1)
    )
    model_order = plot_data[MODEL_SPEC_COLUMN].drop_duplicates().tolist()
    pivoted = plot_data.pivot_table(
        index=MODEL_SPEC_COLUMN,
        columns="precision",
        values="total_average_time",
        aggfunc="mean",
        sort=False,
    ).reindex(model_order)

    figure_width = max(10.0, 1.7 * len(model_order))
    ax = pivoted.plot(kind="bar", figsize=(figure_width, 6), width=0.8)
    ax.set_xlabel(
        "Model specification (d_model, d_ff, context_len, batch_size, layers, heads)"
    )
    ax.set_ylabel(f"Forward + backward + optimizer average time ({time_unit})")
    ax.set_title("Total Average Training Time by Model Specification and Precision")
    ax.tick_params(axis="x", labelrotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Precision")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    figure = ax.get_figure()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.decimals < 0:
        raise ValueError("--decimals must be non-negative")

    summary = build_summary(args.log_dir.resolve(), args.time_unit)

    plot_output_path = args.plot_output.resolve()
    plot_total_average_time(summary, plot_output_path, args.time_unit)

    rounded_summary = summary.round(
        {
            "avg_forward_time": args.decimals,
            "avg_backward_time": args.decimals,
            "avg_optimizer_time": args.decimals,
        }
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rounded_summary.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(rounded_summary.to_string(index=False))
    print(f"\nTiming unit: {args.time_unit}")
    print(f"Saved {len(rounded_summary)} rows to: {output_path}")
    print(f"Saved comparison plot to: {plot_output_path}")


if __name__ == "__main__":
    main()
