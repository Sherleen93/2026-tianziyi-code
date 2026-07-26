"""Summarize the eight data/18 mip_revised.py solver logs."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "数据" / "18"
RESULTS_ROOT = ROOT / "批量求解结果_数据18_20260723"

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def last_match(pattern: str, text: str) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    return matches[-1] if matches else None


def optional_number(value: str) -> float | None:
    return None if value == "-" else float(value)


def parse_case(case_dir: Path, wall_seconds: str) -> dict[str, object]:
    log_path = RESULTS_ROOT / f"{case_dir.name}.log"
    text = log_path.read_text(encoding="utf-8", errors="replace")

    model_match = last_match(
        r"Optimize a model with (\d+) rows, (\d+) columns", text
    )
    explored_match = last_match(
        rf"Explored (\d+) nodes .* in ({NUMBER}) seconds", text
    )
    solution_match = last_match(r"Solution count (\d+)", text)
    best_match = last_match(
        rf"Best objective ({NUMBER}|-), best bound ({NUMBER}|-), "
        rf"gap ({NUMBER}%|-)",
        text,
    )

    if "Optimal solution found" in text:
        status = "OPTIMAL"
    elif "Time limit reached" in text:
        solution_count = int(solution_match.group(1)) if solution_match else 0
        status = (
            "TIME_LIMIT_FEASIBLE"
            if solution_count > 0
            else "TIME_LIMIT_NO_SOLUTION"
        )
    elif "Model is infeasible" in text:
        status = "INFEASIBLE"
    else:
        status = "UNKNOWN"

    orders = read_rows(case_dir / "orders.csv")
    details = read_rows(case_dir / "order_details.csv")
    products = read_rows(case_dir / "products.csv")
    summary = read_rows(case_dir / "dataset_summary.csv")[0]
    sku_to_zone = {
        int(row["sku_id"]): int(row["zone_id"]) for row in products
    }
    zone_picks = {1: 0, 2: 0, 3: 0}
    for detail in details:
        zone = sku_to_zone[int(detail["sku_id"])]
        zone_picks[zone] = zone_picks.get(zone, 0) + 1

    objective = None
    best_bound = None
    gap_percent = None
    if best_match:
        objective = optional_number(best_match.group(1))
        best_bound = optional_number(best_match.group(2))
        gap_text = best_match.group(3)
        gap_percent = None if gap_text == "-" else float(gap_text.rstrip("%"))

    return {
        "case": case_dir.name,
        "status": status,
        "objective": objective,
        "best_bound": best_bound,
        "gap_percent": gap_percent,
        "solve_seconds": (
            float(explored_match.group(2)) if explored_match else None
        ),
        "wall_seconds": float(wall_seconds),
        "nodes": int(explored_match.group(1)) if explored_match else None,
        "solution_count": (
            int(solution_match.group(1)) if solution_match else None
        ),
        "variables": int(model_match.group(2)) if model_match else None,
        "initial_constraints": (
            int(model_match.group(1)) if model_match else None
        ),
        "num_orders": len(orders),
        "num_emergency": int(summary["num_emergency_orders"]),
        "actual_alpha": float(summary["actual_emergency_ratio"]),
        "order_details": len(details),
        "zone1_picks": zone_picks.get(1, 0),
        "zone2_picks": zone_picks.get(2, 0),
        "zone3_picks": zone_picks.get(3, 0),
        "log": log_path.name,
    }


def main() -> None:
    status_rows = read_rows(RESULTS_ROOT / "run_status.csv")
    wall_by_case = {row["case"]: row["wall_seconds"] for row in status_rows}
    case_dirs = sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())
    rows = [
        parse_case(case_dir, wall_by_case[case_dir.name])
        for case_dir in case_dirs
    ]

    output_path = RESULTS_ROOT / "summary.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['case']}: status={row['status']}, "
            f"objective={row['objective']}, bound={row['best_bound']}, "
            f"solve={row['solve_seconds']}s"
        )
    print(f"Summary: {output_path}")


if __name__ == "__main__":
    main()
