#!/usr/bin/env python3
"""Run compute_zone_lower_bound_7_23.py for all eight 15-order instances."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from compute_zone_lower_bound_7_23 import (
    A,
    PI,
    SIGMA,
    compute_lower_bound,
    load_lower_bound_data,
    save_details,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parents[1]
DATA_ROOT = (
    WORKSPACE
    / "2026-tianziyi-Gurobi-7-13_check"
    / "原版-备份-不修改"
    / "Gurobi"
    / "数据"
    / "15"
)
OUTPUT_ROOT = SCRIPT_DIR / "lower_bound_results_data15_7_23"
SUMMARY_PATH = OUTPUT_ROOT / "all_8_instances_lower_bound_summary_7_23.csv"
TIMING_PATH = OUTPUT_ROOT / "batch_timing_7_23.csv"

INSTANCE_PATTERN = re.compile(
    r"^Medical_mu(?P<mu>\d+)_beta(?P<beta>\d+)_alpha0(?P<alpha>\d+)"
    r"_rack(?P<rack>\d+)$"
)


def parse_instance_name(name: str) -> dict[str, int]:
    match = INSTANCE_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unexpected instance directory name: {name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def main() -> None:
    required_files = {"orders.csv", "order_details.csv", "products.csv"}
    instance_dirs = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    if len(instance_dirs) != 8:
        raise RuntimeError(
            f"Expected 8 instance directories in {DATA_ROOT}, found {len(instance_dirs)}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    batch_start = time.perf_counter()

    for instance_dir in instance_dirs:
        missing = sorted(
            filename
            for filename in required_files
            if not (instance_dir / filename).is_file()
        )
        if missing:
            raise FileNotFoundError(
                f"{instance_dir.name} is missing required files: {missing}"
            )

        case_start = time.perf_counter()

        load_start = time.perf_counter()
        order_info, order_needs, zones = load_lower_bound_data(instance_dir)
        load_seconds = time.perf_counter() - load_start

        compute_start = time.perf_counter()
        result = compute_lower_bound(
            order_info=order_info,
            order_needs=order_needs,
            zones=zones,
            sigma=SIGMA,
            a=A,
            pi=PI,
        )
        compute_seconds = time.perf_counter() - compute_start

        save_start = time.perf_counter()
        save_details(result, OUTPUT_ROOT / instance_dir.name)
        save_seconds = time.perf_counter() - save_start
        total_seconds = time.perf_counter() - case_start

        cli_start = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "compute_zone_lower_bound_7_23.py"),
                "--data-dir",
                str(instance_dir),
                "--output-dir",
                str(OUTPUT_ROOT / instance_dir.name),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        cli_wall_seconds = time.perf_counter() - cli_start

        cli_summary = pd.read_csv(
            OUTPUT_ROOT / instance_dir.name / "lower_bound_summary_7_23.csv"
        ).iloc[0]
        if abs(float(cli_summary["lower_bound"]) - result["lower_bound"]) > 1e-9:
            raise RuntimeError(
                f"CLI result mismatch for {instance_dir.name}: "
                f"{cli_summary['lower_bound']} != {result['lower_bound']}"
            )

        params = parse_instance_name(instance_dir.name)
        binding_zones = ",".join(map(str, result["binding_zones"]))
        summary_rows.append(
            {
                "instance": instance_dir.name,
                **params,
                "number_of_orders": len(order_info),
                "number_of_zones": len(zones),
                "lower_bound": result["lower_bound"],
                "baseline_bound": result["baseline_bound"],
                "selected_increment": result["selected_increment"],
                "total_weight": result["total_weight"],
                "binding_zones": binding_zones,
                "q": result["zone_capacity_per_slot"],
                "pi": PI,
                "sigma": SIGMA,
                "a": A,
                "load_seconds": load_seconds,
                "compute_seconds": compute_seconds,
                "save_seconds": save_seconds,
                "total_seconds": total_seconds,
                "cli_wall_seconds": cli_wall_seconds,
            }
        )

        print(
            f"{instance_dir.name}: LB={result['lower_bound']:.10g}, "
            f"compute={compute_seconds:.6f}s, "
            f"in-process total={total_seconds:.6f}s, "
            f"CLI wall={cli_wall_seconds:.6f}s"
        )

    batch_seconds = time.perf_counter() - batch_start
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    pd.DataFrame(
        [
            {
                "number_of_instances": len(summary_rows),
                "batch_total_seconds": batch_seconds,
                "sum_case_total_seconds": summary["total_seconds"].sum(),
                "sum_compute_seconds": summary["compute_seconds"].sum(),
                "sum_cli_wall_seconds": summary["cli_wall_seconds"].sum(),
            }
        ]
    ).to_csv(TIMING_PATH, index=False)

    print(f"Summary: {SUMMARY_PATH}")
    print(f"Timing : {TIMING_PATH}")
    print(f"Batch total: {batch_seconds:.6f}s")


if __name__ == "__main__":
    main()
