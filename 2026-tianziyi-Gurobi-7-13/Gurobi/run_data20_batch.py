"""Run all eight 20-order instances with mip_revised.py in parallel."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "数据" / "20"
RESULTS_ROOT = ROOT / "批量求解结果_数据20_20260723"
SOLVER = ROOT / "mip_revised.py"
WRAPPER = ROOT / "run_mip_revised_size_limited_lazy.py"
MAX_PARALLEL_CASES = 8
THREADS_PER_CASE = 2


def run_case(case_dir: Path) -> dict[str, object]:
    case_name = case_dir.name
    log_path = RESULTS_ROOT / f"{case_name}.log"
    environment = os.environ.copy()
    environment["GUROBI_THREADS"] = str(THREADS_PER_CASE)

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            [sys.executable, str(WRAPPER), str(SOLVER)],
            cwd=case_dir,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    return {
        "case": case_name,
        "exit_code": completed.returncode,
        "wall_seconds": f"{time.monotonic() - started:.3f}",
        "log": log_path.name,
    }


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    if len(case_dirs) != 8:
        raise RuntimeError(f"Expected 8 cases under {DATA_ROOT}, found {len(case_dirs)}")

    print(
        f"Running {len(case_dirs)} cases with {MAX_PARALLEL_CASES} parallel "
        f"processes and {THREADS_PER_CASE} Gurobi threads per case.",
        flush=True,
    )
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_CASES) as executor:
        futures = {executor.submit(run_case, path): path.name for path in case_dirs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(
                f"DONE {row['case']} exit={row['exit_code']} "
                f"wall={row['wall_seconds']}s",
                flush=True,
            )

    rows.sort(key=lambda row: str(row["case"]))
    with (RESULTS_ROOT / "run_status.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"ALL_DONE {RESULTS_ROOT}", flush=True)


if __name__ == "__main__":
    main()
