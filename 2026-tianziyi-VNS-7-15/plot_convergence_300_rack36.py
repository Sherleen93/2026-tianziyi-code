#!/usr/bin/env python3
"""
Run and plot the 300/rack36 convergence experiment.

This script is an orchestration layer only. It reuses the existing project
implementations of:

- VNS-DPH, SA-DPH and TS-DPH from
  三个启发式代码/vns_sa_ts_compare_multi_scales.py
- FCFS from
  FCFS代码/large_scale_fcfs_vns_greedy_vns_dph_clean.py

It does not reimplement the optimization algorithms or their objective
function. The three stochastic algorithms are run sequentially for each seed
so that concurrent processes do not compete for the same CPU-time budget.
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator


ALGORITHM_LABELS = {
    "VNS": "VNS-DPH",
    "SA": "SA-DPH",
    "TS": "TS-DPH",
    "FCFS": "FCFS",
}

STOCHASTIC_ALGORITHMS = ("VNS", "SA", "TS")

PLOT_STYLES = {
    "VNS": {
        "color": "#4A7DBA",
        "linestyle": "-",
        "marker": "o",
        "zorder": 5,
    },
    "SA": {
        "color": "#8FC7DE",
        "linestyle": "-",
        "marker": "^",
        "zorder": 4,
    },
    "TS": {
        "color": "#EC6343",
        "linestyle": "-",
        "marker": "D",
        "zorder": 3,
    },
    "FCFS": {
        "color": "#F5B46F",
        "linestyle": "-",
        "marker": "s",
        "zorder": 2,
    },
}

instance_size=300
order_count=300

@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_path: Path
    output_directory: Path
    stochastic_module_path: Path
    fcfs_module_path: Path


def parse_seeds(values: Sequence[str] | None) -> list[int]:
    """Parse seeds supplied either as separate values or comma-separated text."""
    if not values:
        return [0, 1, 2, 3, 4]

    parsed: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                parsed.append(int(token))

    if not parsed:
        raise ValueError("At least one random seed is required.")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate seeds are not allowed: {parsed}")
    return parsed


def import_module_from_path(module_path: Path, module_name: str) -> ModuleType:
    """Import a Python source file without modifying sys.path."""
    if not module_path.is_file():
        raise FileNotFoundError(f"Algorithm module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import specification for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def require_callable(module: ModuleType, name: str) -> None:
    """Fail early when an expected project API is missing."""
    value = getattr(module, name, None)
    if not callable(value):
        raise AttributeError(
            f"Required callable {name!r} was not found in {module.__file__}"
        )


def validate_csv_columns(csv_path: Path, required_columns: set[str]) -> pd.DataFrame:
    """Read one input CSV and validate its required columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{csv_path.name} contains no data rows.")
    return frame


def validate_instance_files(data_path: Path) -> None:
    """Validate the expected 300-order, 36-rack dataset structure."""
    orders = validate_csv_columns(
        data_path / "orders.csv",
        {"order_id", "type", "priority", "weight"},
    )
    details = validate_csv_columns(
        data_path / "order_details.csv",
        {"order_id", "sku_id"},
    )
    products = validate_csv_columns(
        data_path / "products.csv",
        {"sku_id", "zone_id"},
    )
    inventory = validate_csv_columns(
        data_path / "inventory.csv",
        {"rack_id", "sku_id", "zone_id"},
    )

    order_count = int(orders["order_id"].nunique())
    rack_count = int(inventory["rack_id"].nunique())
    if order_count != 300:
        raise ValueError(
            f"Expected 300 unique orders, but found {order_count} in {data_path}."
        )
    if rack_count != 36:
        raise ValueError(
            f"Expected 36 unique racks, but found {rack_count} in {data_path}."
        )

    order_ids = set(orders["order_id"])
    unknown_orders = set(details["order_id"]).difference(order_ids)
    if unknown_orders:
        preview = sorted(unknown_orders)[:10]
        raise ValueError(
            "order_details.csv references order IDs absent from orders.csv: "
            f"{preview}"
        )

    product_ids = set(products["sku_id"])
    unknown_detail_skus = set(details["sku_id"]).difference(product_ids)
    unknown_inventory_skus = set(inventory["sku_id"]).difference(product_ids)
    if unknown_detail_skus:
        preview = sorted(unknown_detail_skus)[:10]
        raise ValueError(
            "order_details.csv references SKU IDs absent from products.csv: "
            f"{preview}"
        )
    if unknown_inventory_skus:
        preview = sorted(unknown_inventory_skus)[:10]
        raise ValueError(
            "inventory.csv references SKU IDs absent from products.csv: "
            f"{preview}"
        )


def load_instance(paths: ProjectPaths) -> tuple[ModuleType, ModuleType]:
    """
    Validate the dataset and load the two existing project algorithm modules.

    The project modules keep their instance data in module-level globals, so
    their existing initialize_data functions are called before each run.
    """
    validate_instance_files(paths.data_path)

    stochastic_module = import_module_from_path(
        paths.stochastic_module_path,
        "project_vns_sa_ts_300_rack36",
    )
    fcfs_module = import_module_from_path(
        paths.fcfs_module_path,
        "project_fcfs_300_rack36",
    )

    for name in (
        "initialize_data",
        "vns_optimize",
        "sa_optimize",
        "ts_optimize",
    ):
        require_callable(stochastic_module, name)

    for name in ("initialize_data", "fcfs_greedy_optimize"):
        require_callable(fcfs_module, name)

    if not stochastic_module.initialize_data(
        str(paths.data_path), instance_size=instance_size
    ):
        raise RuntimeError(
            f"The VNS/SA/TS module could not initialize {paths.data_path}."
        )
    if not fcfs_module.initialize_data(str(paths.data_path), instance_size=instance_size):
        raise RuntimeError(
            f"The FCFS module could not initialize {paths.data_path}."
        )

    return stochastic_module, fcfs_module


def set_random_seed(seed: int) -> None:
    """Set the standard-library and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def run_algorithm(
    algorithm: str,
    seed: int,
    time_limit: float,
    data_path: Path,
    stochastic_module: ModuleType,
    fcfs_module: ModuleType,
) -> dict[str, Any]:
    """Call one existing project algorithm entry point."""
    set_random_seed(seed)

    if algorithm in STOCHASTIC_ALGORITHMS:
        if not stochastic_module.initialize_data(str(data_path), instance_size=instance_size):
            raise RuntimeError(
                f"Failed to reinitialize stochastic data for {algorithm}, seed {seed}."
            )
        optimizer = {
            "VNS": stochastic_module.vns_optimize,
            "SA": stochastic_module.sa_optimize,
            "TS": stochastic_module.ts_optimize,
        }[algorithm]
    elif algorithm == "FCFS":
        if not fcfs_module.initialize_data(str(data_path), instance_size=instance_size):
            raise RuntimeError("Failed to reinitialize data for FCFS.")
        optimizer = fcfs_module.fcfs_greedy_optimize
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    wall_start = time.perf_counter()
    result = optimizer(
        time_limit_seconds=time_limit,
        quiet=True,
        random_seed=seed,
    )
    measured_runtime = time.perf_counter() - wall_start

    if not isinstance(result, dict):
        raise TypeError(
            f"{algorithm} returned {type(result).__name__}, expected dict."
        )

    final_objective = result.get("best_cost")
    if final_objective is None or not np.isfinite(float(final_objective)):
        raise RuntimeError(
            f"{algorithm}, seed {seed} did not return a finite best_cost."
        )

    result = dict(result)
    result.setdefault("seed", seed)
    result.setdefault("elapsed_seconds", measured_runtime)
    result["_measured_wall_runtime"] = measured_runtime
    return result


def collect_convergence_history(
    result: dict[str, Any],
    algorithm: str,
    seed: int | str,
    time_limit: float,
) -> list[dict[str, Any]]:
    """
    Extract improvement events from the project's iteration_records.

    Only the best-so-far objective is retained. For a minimization problem,
    every retained point must be no larger than the previous point.
    """
    final_objective = float(result["best_cost"])

    if algorithm == "FCFS":
        return [
            {
                "algorithm": ALGORITHM_LABELS[algorithm],
                "seed": seed,
                "elapsed_time": 0.0,
                "best_objective": final_objective,
                "improvement_index": 0,
            }
        ]

    records = result.get("iteration_records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(
            f"{algorithm}, seed {seed} returned no iteration_records. "
            "The current project implementation is expected to return records "
            "containing elapsed_seconds and best_cost. If that interface is "
            "removed in a future version, add a non-invasive optional history "
            "callback to the algorithm module before using this script."
        )

    normalized: list[tuple[float, float]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"{algorithm}, seed {seed}, iteration record {index} is not a dict."
            )

        elapsed = record.get("elapsed_seconds")
        best_cost = record.get("best_cost")
        if elapsed is None or best_cost is None:
            raise KeyError(
                f"{algorithm}, seed {seed}, iteration record {index} must contain "
                "elapsed_seconds and best_cost."
            )

        elapsed_value = float(elapsed)
        best_value = float(best_cost)
        if not np.isfinite(elapsed_value) or not np.isfinite(best_value):
            raise ValueError(
                f"{algorithm}, seed {seed}, iteration record {index} contains "
                "a non-finite elapsed time or objective."
            )

        normalized.append(
            (
                min(max(elapsed_value, 0.0), time_limit),
                best_value,
            )
        )

    normalized.sort(key=lambda item: item[0])
    if normalized[0][0] > 0.0:
        normalized.insert(0, (0.0, normalized[0][1]))

    improvement_events: list[dict[str, Any]] = []
    incumbent = float("inf")
    for elapsed_value, reported_best in normalized:
        incumbent = min(incumbent, reported_best)
        if (
            not improvement_events
            or incumbent < float(improvement_events[-1]["best_objective"])
        ):
            improvement_events.append(
                {
                    "algorithm": ALGORITHM_LABELS[algorithm],
                    "seed": seed,
                    "elapsed_time": elapsed_value,
                    "best_objective": incumbent,
                    "improvement_index": len(improvement_events),
                }
            )

    if final_objective < float(improvement_events[-1]["best_objective"]):
        final_elapsed = min(
            max(float(result.get("elapsed_seconds", time_limit)), 0.0),
            time_limit,
        )
        improvement_events.append(
            {
                "algorithm": ALGORITHM_LABELS[algorithm],
                "seed": seed,
                "elapsed_time": final_elapsed,
                "best_objective": final_objective,
                "improvement_index": len(improvement_events),
            }
        )

    # Coalesce multiple improvements at the same clipped time, retaining the best.
    coalesced: list[dict[str, Any]] = []
    for event in improvement_events:
        if coalesced and np.isclose(
            float(event["elapsed_time"]),
            float(coalesced[-1]["elapsed_time"]),
        ):
            if float(event["best_objective"]) < float(
                coalesced[-1]["best_objective"]
            ):
                coalesced[-1] = event
        else:
            coalesced.append(event)

    for index, event in enumerate(coalesced):
        event["improvement_index"] = index

    objectives = np.asarray(
        [float(event["best_objective"]) for event in coalesced],
        dtype=float,
    )
    if np.any(np.diff(objectives) > 0.0):
        raise AssertionError(
            f"{algorithm}, seed {seed} produced a non-monotone best-so-far history."
        )
    return coalesced


def make_time_grid(time_limit: float, sampling_interval: float) -> np.ndarray:
    """Create a time grid that always includes both 0 and time_limit."""
    if time_limit <= 0:
        raise ValueError("time_limit must be positive.")
    if sampling_interval <= 0:
        raise ValueError("sampling_interval must be positive.")

    grid = np.arange(
        0.0,
        time_limit + sampling_interval * 0.5,
        sampling_interval,
        dtype=float,
    )
    grid = grid[grid <= time_limit]
    if grid.size == 0 or not np.isclose(grid[0], 0.0):
        grid = np.insert(grid, 0, 0.0)
    if not np.isclose(grid[-1], time_limit):
        grid = np.append(grid, time_limit)
    else:
        grid[-1] = time_limit
    return grid


def convert_to_time_grid(
    raw_frame: pd.DataFrame,
    time_grid: np.ndarray,
) -> pd.DataFrame:
    """Forward-fill each algorithm/seed best-so-far trajectory on the common grid."""
    required = {"algorithm", "seed", "elapsed_time", "best_objective"}
    missing = required.difference(raw_frame.columns)
    if missing:
        raise ValueError(f"Raw history is missing columns: {sorted(missing)}")

    aligned_frames: list[pd.DataFrame] = []
    for (algorithm, seed), group in raw_frame.groupby(
        ["algorithm", "seed"],
        sort=False,
        dropna=False,
    ):
        group = group.sort_values(["elapsed_time", "improvement_index"])
        times = group["elapsed_time"].to_numpy(dtype=float)
        objectives = group["best_objective"].to_numpy(dtype=float)

        if times.size == 0:
            raise ValueError(f"No convergence points for {algorithm}, seed {seed}.")
        if times[0] > 0.0:
            raise ValueError(
                f"The first convergence point for {algorithm}, seed {seed} "
                "must be at time 0."
            )
        if np.any(np.diff(objectives) > 0.0):
            raise ValueError(
                f"Raw best-so-far values increase for {algorithm}, seed {seed}."
            )

        indices = np.searchsorted(times, time_grid, side="right") - 1
        indices = np.clip(indices, 0, len(objectives) - 1)
        aligned_frames.append(
            pd.DataFrame(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "elapsed_time": time_grid,
                    "best_objective": objectives[indices],
                }
            )
        )

    return pd.concat(aligned_frames, ignore_index=True)


def aggregate_across_seeds(grid_frame: pd.DataFrame) -> pd.DataFrame:
    """Average the aligned best-so-far curves across seeds."""
    mean_frame = (
        grid_frame.groupby(
            ["algorithm", "elapsed_time"],
            as_index=False,
            sort=False,
        )
        .agg(
            mean_best_objective=("best_objective", "mean"),
            std_best_objective=("best_objective", "std"),
            min_best_objective=("best_objective", "min"),
            max_best_objective=("best_objective", "max"),
            n_runs=("seed", "nunique"),
        )
        .sort_values(["algorithm", "elapsed_time"], ignore_index=True)
    )
    mean_frame["std_best_objective"] = mean_frame[
        "std_best_objective"
    ].fillna(0.0)
    return mean_frame


def save_raw_results(raw_records: list[dict[str, Any]], output_path: Path) -> None:
    """Save all retained improvement events."""
    raw_frame = pd.DataFrame(raw_records)
    raw_frame["seed"] = raw_frame["seed"].astype(str)
    raw_frame = raw_frame.sort_values(
        ["algorithm", "seed", "elapsed_time", "improvement_index"],
        ignore_index=True,
    )
    raw_frame.to_csv(output_path, index=False, encoding="utf-8")


def build_summary_frame(run_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a per-run summary with repeated algorithm-level statistics."""
    frame = pd.DataFrame(run_rows)
    frame["seed"] = frame["seed"].astype(str)
    statistics = (
        frame.groupby("algorithm", as_index=False, sort=False)
        .agg(
            mean_final_objective=("final_objective", "mean"),
            best_final_objective=("final_objective", "min"),
            std_final_objective=("final_objective", "std"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            n_runs=("seed", "nunique"),
        )
    )
    statistics["std_final_objective"] = statistics[
        "std_final_objective"
    ].fillna(0.0)

    summary = frame.merge(statistics, on="algorithm", how="left")
    return summary[
        [
            "algorithm",
            "seed",
            "final_objective",
            "runtime_seconds",
            "mean_final_objective",
            "best_final_objective",
            "std_final_objective",
            "mean_runtime_seconds",
            "n_runs",
        ]
    ].sort_values(["algorithm", "seed"], ignore_index=True)


def configure_plot_style() -> None:
    """Set a restrained English-journal plotting style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def plot_convergence(
    mean_frame: pd.DataFrame,
    output_directory: Path,
    figure_width: float,
    figure_height: float,
) -> None:
    """Create the four requested convergence curves and export all formats."""
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    plot_order = ("VNS-DPH", "SA-DPH", "TS-DPH", "FCFS")
    label_to_key = {label: key for key, label in ALGORITHM_LABELS.items()}

    for label in plot_order:
        algorithm_key = label_to_key[label]
        subset = mean_frame.loc[mean_frame["algorithm"] == label].sort_values(
            "elapsed_time"
        )
        if subset.empty:
            raise ValueError(f"No mean convergence data found for {label}.")

        style = PLOT_STYLES[algorithm_key]
        marker_stride = max(1, len(subset) // 12)
        ax.plot(
            subset["elapsed_time"],
            subset["mean_best_objective"],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.6,
            drawstyle="steps-post",
            marker=style["marker"],
            markersize=4.0,
            markevery=marker_stride,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=style["zorder"],
        )

    y_values = mean_frame["mean_best_objective"].to_numpy(dtype=float)
    if not np.isfinite(y_values).all():
        raise ValueError("The mean convergence data contains non-finite values.")
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_span = max(y_max - y_min, 1.0)

    ax.set_xlim(
        float(mean_frame["elapsed_time"].min()),
        float(mean_frame["elapsed_time"].max()),
    )
    ax.set_ylim(y_min - 0.06 * y_span, y_max + 0.06 * y_span)
    ax.set_xlabel("Running time (s)")
    ax.set_ylabel("Total weighted completion time")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(True, color="#D9D9D9", linewidth=0.6, alpha=0.55)
    ax.tick_params(axis="both", width=0.7, length=3)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.tight_layout()
    output_stem = output_directory / "convergence_300_rack36"
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        output_stem.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def print_terminal_means(summary_frame: pd.DataFrame) -> None:
    """Print the final mean objective for each requested algorithm."""
    print("\nFinal objective means")
    print("-" * 48)
    for algorithm in ("VNS-DPH", "SA-DPH", "TS-DPH", "FCFS"):
        subset = summary_frame.loc[summary_frame["algorithm"] == algorithm]
        if subset.empty:
            raise ValueError(f"Missing summary rows for {algorithm}.")
        value = float(subset["mean_final_objective"].iloc[0])
        print(f"{algorithm:<10s} {value:,.3f}")


def build_argument_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing project algorithms on 300/rack36 and create "
            "four convergence curves without a lower bound."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=project_root / "300" / "rack36",
        help="Directory containing orders.csv, order_details.csv, products.csv and inventory.csv.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project_root / "results" / "convergence_300_rack36",
        help="New output directory; existing result files are not overwritten unless --overwrite is used.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Time limit in seconds for each algorithm run (default: 300).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["0", "1", "2", "3", "4"],
        help="Random seeds as separate values or comma-separated text.",
    )
    parser.add_argument(
        "--sampling-interval",
        type=float,
        default=1.0,
        help="Common time-grid interval in seconds (default: 1).",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=7.2,
        help="Figure width in inches (default: 7.2, suitable for a double column).",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=4.6,
        help="Figure height in inches (default: 4.6).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of result files in the output directory.",
    )
    return parser


def prepare_output_directory(output_directory: Path, overwrite: bool) -> None:
    """Create a new output directory and guard against accidental overwrites."""
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_files = (
        "convergence_300_rack36.png",
        "convergence_300_rack36.tiff",
        "convergence_300_rack36.pdf",
        "convergence_300_rack36.svg",
        "convergence_300_rack36_raw.csv",
        "convergence_300_rack36_mean.csv",
        "convergence_300_rack36_summary.csv",
    )
    existing = [
        output_directory / filename
        for filename in expected_files
        if (output_directory / filename).exists()
    ]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise FileExistsError(
            "The following output files already exist:\n"
            f"{formatted}\n"
            "Use a different --output-directory or pass --overwrite."
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    parser = build_argument_parser(project_root)
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    paths = ProjectPaths(
        project_root=project_root,
        data_path=args.data_path.expanduser().resolve(),
        output_directory=args.output_directory.expanduser().resolve(),
        stochastic_module_path=(
            project_root
            / "三个启发式代码"
            / "vns_sa_ts_compare_multi_scales.py"
        ),
        fcfs_module_path=(
            project_root
            / "FCFS代码"
            / "large_scale_fcfs_vns_greedy_vns_dph_clean.py"
        ),
    )

    prepare_output_directory(paths.output_directory, args.overwrite)
    stochastic_module, fcfs_module = load_instance(paths)
    time_grid = make_time_grid(args.time_limit, args.sampling_interval)

    raw_records: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for algorithm in STOCHASTIC_ALGORITHMS:
        for run_number, seed in enumerate(seeds, start=1):
            print(
                f"[{ALGORITHM_LABELS[algorithm]}] "
                f"run {run_number}/{len(seeds)}, seed={seed}, "
                f"time_limit={args.time_limit:g}s",
                flush=True,
            )
            result = run_algorithm(
                algorithm=algorithm,
                seed=seed,
                time_limit=args.time_limit,
                data_path=paths.data_path,
                stochastic_module=stochastic_module,
                fcfs_module=fcfs_module,
            )
            events = collect_convergence_history(
                result=result,
                algorithm=algorithm,
                seed=seed,
                time_limit=args.time_limit,
            )
            raw_records.extend(events)
            run_rows.append(
                {
                    "algorithm": ALGORITHM_LABELS[algorithm],
                    "seed": seed,
                    "final_objective": float(result["best_cost"]),
                    "runtime_seconds": float(result["elapsed_seconds"]),
                }
            )

    # The current project FCFS entry point is deterministic, so it is run once.
    print("[FCFS] deterministic single run", flush=True)
    fcfs_seed = seeds[0]
    fcfs_result = run_algorithm(
        algorithm="FCFS",
        seed=fcfs_seed,
        time_limit=args.time_limit,
        data_path=paths.data_path,
        stochastic_module=stochastic_module,
        fcfs_module=fcfs_module,
    )
    raw_records.extend(
        collect_convergence_history(
            result=fcfs_result,
            algorithm="FCFS",
            seed="deterministic",
            time_limit=args.time_limit,
        )
    )
    run_rows.append(
        {
            "algorithm": "FCFS",
            "seed": "deterministic",
            "final_objective": float(fcfs_result["best_cost"]),
            "runtime_seconds": float(fcfs_result["elapsed_seconds"]),
        }
    )

    raw_path = (
        paths.output_directory / "convergence_300_rack36_raw.csv"
    )
    save_raw_results(raw_records, raw_path)
    raw_frame = pd.read_csv(raw_path)

    grid_frame = convert_to_time_grid(raw_frame, time_grid)
    mean_frame = aggregate_across_seeds(grid_frame)
    mean_frame.to_csv(
        paths.output_directory / "convergence_300_rack36_mean.csv",
        index=False,
        encoding="utf-8",
    )

    summary_frame = build_summary_frame(run_rows)
    summary_frame.to_csv(
        paths.output_directory / "convergence_300_rack36_summary.csv",
        index=False,
        encoding="utf-8",
    )

    plot_convergence(
        mean_frame=mean_frame,
        output_directory=paths.output_directory,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
    )
    print_terminal_means(summary_frame)
    print(f"\nOutputs written to: {paths.output_directory}")


if __name__ == "__main__":
    main()
