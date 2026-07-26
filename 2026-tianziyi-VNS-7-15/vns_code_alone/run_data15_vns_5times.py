#!/usr/bin/env python3
"""Run the standard VNS implementation five times on all 15-order instances."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from types import ModuleType

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
ALGORITHM_PATH = SCRIPT_DIR / "vns_dp_iteration_standard_vns.py"
DATA_ROOT = (
    WORKSPACE
    / "2026-tianziyi-Gurobi-7-13_check"
    / "原版-备份-不修改"
    / "Gurobi"
    / "数据"
    / "15"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "vns_results_data15_5runs_20260723"
SEEDS = (1, 2, 3, 4, 5)
MAX_ITERATIONS = 30

INSTANCE_PATTERN = re.compile(
    r"^Medical_mu(?P<mu>\d+)_beta(?P<beta>\d+)_alpha0(?P<alpha>\d+)"
    r"_rack(?P<rack>\d+)$"
)


def load_algorithm() -> ModuleType:
    spec = importlib.util.spec_from_file_location("standard_vns_algorithm", ALGORITHM_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load algorithm from {ALGORITHM_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def activate_instance(module: ModuleType, instance_dir: Path) -> int:
    previous_cwd = Path.cwd()
    try:
        os.chdir(instance_dir)
        data = module.load_data()
    finally:
        os.chdir(previous_cwd)

    if data[0] is None:
        raise RuntimeError(f"Failed to load data from {instance_dir}")

    module.DATA = data
    (
        module.ORDER_INFO,
        module.ORDER_NEEDS_MASTER,
        module.RACK_INV,
        module.ZONES,
        module.ORDER_TOTAL_DEMANDS,
    ) = data

    module.ORDER_SKU_CACHE = {}
    module.ORDER_RACK_SKU_CACHE = {}
    module.ORDER_RACKS_CACHE = {}
    for order_id, zones_dict in module.ORDER_NEEDS_MASTER.items():
        module.ORDER_SKU_CACHE[order_id] = set().union(*zones_dict.values())
        racks = set()
        for zone, skus in zones_dict.items():
            for sku in skus:
                for rack_id, rack_skus in module.RACK_INV[zone].items():
                    if sku in rack_skus:
                        racks.add((zone, rack_id))
        module.ORDER_RACKS_CACHE[order_id] = racks

    return len(module.ORDER_INFO)


def parse_instance_name(name: str) -> dict[str, int]:
    match = INSTANCE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected instance name: {name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def write_summary(raw_runs: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    successful = raw_runs.loc[raw_runs["status"] == "ok"].copy()
    summary = (
        successful.groupby(
            [
                "instance",
                "mu",
                "beta",
                "alpha",
                "rack",
                "number_of_orders",
            ],
            as_index=False,
        )
        .agg(
            completed_runs=("best_cost", "count"),
            average_best_cost=("best_cost", "mean"),
            std_best_cost=("best_cost", "std"),
            best_of_5=("best_cost", "min"),
            worst_of_5=("best_cost", "max"),
            average_runtime_seconds=("runtime_seconds", "mean"),
            std_runtime_seconds=("runtime_seconds", "std"),
            average_dph_evaluations=("dph_evaluations", "mean"),
        )
        .sort_values("instance")
    )
    summary.to_csv(
        output_root / "all_8_instances_vns_5runs_summary.csv", index=False
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--instance",
        action="append",
        help="Run only the named instance; may be supplied more than once.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.runs > len(SEEDS):
        raise ValueError(f"--runs must be between 1 and {len(SEEDS)}")

    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    instance_dirs = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    if args.instance:
        selected = set(args.instance)
        instance_dirs = [path for path in instance_dirs if path.name in selected]
        missing = selected - {path.name for path in instance_dirs}
        if missing:
            raise ValueError(f"Unknown instances: {sorted(missing)}")
    elif len(instance_dirs) != 8:
        raise RuntimeError(f"Expected 8 instances, found {len(instance_dirs)}")

    module = load_algorithm()
    records: list[dict[str, object]] = []
    raw_path = output_root / "all_runs_raw.csv"
    batch_start = time.perf_counter()

    for instance_dir in instance_dirs:
        number_of_orders = activate_instance(module, instance_dir)
        params = parse_instance_name(instance_dir.name)

        for run_number, seed in enumerate(SEEDS[: args.runs], start=1):
            run_start = time.perf_counter()
            try:
                result = module.vns_optimize(
                    max_iterations=MAX_ITERATIONS,
                    quiet=True,
                    random_seed=seed,
                    vns_ride_share_window=module.VNS_RIDE_SHARE_WINDOW,
                    vns_ride_share_require=module.VNS_RIDE_SHARE_REQUIRE,
                )
                runtime_seconds = time.perf_counter() - run_start
                run_dir = output_root / instance_dir.name
                run_dir.mkdir(parents=True, exist_ok=True)
                module.save_iteration_records(
                    result["iteration_records"],
                    str(run_dir / f"run_{run_number}_seed_{seed}_iteration_history.csv"),
                )
                record = {
                    "instance": instance_dir.name,
                    **params,
                    "number_of_orders": number_of_orders,
                    "run": run_number,
                    "random_seed": seed,
                    "status": "ok",
                    "initial_cost": result["initial_cost"],
                    "best_cost": result["best_cost"],
                    "runtime_seconds": runtime_seconds,
                    "algorithm_elapsed_seconds": result["elapsed_seconds"],
                    "actual_iterations": result["actual_iterations"],
                    "dph_evaluations": result["dph_evaluations"],
                    "last_improvement_iteration": result[
                        "last_improvement_iteration"
                    ],
                    "best_sequence": json.dumps(result["best_seq"]),
                    "error": "",
                }
                print(
                    f"{instance_dir.name} run {run_number}/"
                    f"{args.runs} seed={seed}: best={result['best_cost']:.10g}, "
                    f"time={runtime_seconds:.6f}s",
                    flush=True,
                )
            except Exception as exc:
                runtime_seconds = time.perf_counter() - run_start
                record = {
                    "instance": instance_dir.name,
                    **params,
                    "number_of_orders": number_of_orders,
                    "run": run_number,
                    "random_seed": seed,
                    "status": "error",
                    "initial_cost": None,
                    "best_cost": None,
                    "runtime_seconds": runtime_seconds,
                    "algorithm_elapsed_seconds": None,
                    "actual_iterations": None,
                    "dph_evaluations": None,
                    "last_improvement_iteration": None,
                    "best_sequence": "",
                    "error": repr(exc),
                }
                print(
                    f"{instance_dir.name} run {run_number}/{args.runs} "
                    f"seed={seed}: ERROR {exc!r}",
                    flush=True,
                )

            records.append(record)
            pd.DataFrame(records).to_csv(raw_path, index=False)

    batch_seconds = time.perf_counter() - batch_start
    raw_runs = pd.DataFrame(records)
    summary = write_summary(raw_runs, output_root)
    pd.DataFrame(
        [
            {
                "instances": len(instance_dirs),
                "runs_per_instance": args.runs,
                "successful_runs": int((raw_runs["status"] == "ok").sum()),
                "failed_runs": int((raw_runs["status"] != "ok").sum()),
                "batch_wall_seconds": batch_seconds,
                "sum_vns_runtime_seconds": raw_runs.loc[
                    raw_runs["status"] == "ok", "runtime_seconds"
                ].sum(),
            }
        ]
    ).to_csv(output_root / "batch_timing.csv", index=False)

    print(summary.to_string(index=False), flush=True)
    print(f"Raw results: {raw_path}", flush=True)
    print(f"Batch wall time: {batch_seconds:.6f}s", flush=True)


if __name__ == "__main__":
    main()
