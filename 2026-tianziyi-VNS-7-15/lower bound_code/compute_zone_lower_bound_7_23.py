"""Compute the analytical lower bound defined in manuscript Section 3.4.

Updated on 2026-07-23.

For every non-empty zone p, the calculation is

    q       = floor(sigma / a)
    u_op    = d_op / q
    Z_p     = sum_o w_o * C_op       (WSPT order: u_op / w_o)
    LB_p    = pi * W + sigma * (Z_p - W_p)

The global lower bound is

    LB = pi * W + sigma * max(0, max_p(Z_p - W_p)).

Zone values are deliberately not summed because a split order contributes only
once to the original objective.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Hashable, List, Mapping, Sequence, Set

import pandas as pd


# Parameters used by the VNS-DPH experiments and manuscript.
PI = 120.0      # Completion-time offset.
SIGMA = 20.0   # Length of one synchronized picking slot.
A = 10.0       # Unit SKU-picking time.


OrderId = Hashable
ZoneId = Hashable
SkuId = Hashable


def _python_scalar(value: Any) -> Any:
    """Convert NumPy scalar identifiers to ordinary Python scalars."""
    return value.item() if hasattr(value, "item") else value


def _stable_id_key(value: Any) -> tuple[str, str]:
    """Provide deterministic ordering for numeric or string identifiers."""
    return type(value).__name__, str(value)


def _require_columns(df: pd.DataFrame, columns: Set[str], filename: str) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"{filename} is missing required column(s): {sorted(missing)}"
        )


def load_lower_bound_data(
    data_dir: Path | str,
) -> tuple[
    Dict[OrderId, Dict[str, Any]],
    Dict[OrderId, Dict[ZoneId, Set[SkuId]]],
    List[ZoneId],
]:
    """Load the three CSV files required by the Section 3.4 relaxation.

    Required files are ``orders.csv``, ``order_details.csv``, and
    ``products.csv``. ``inventory.csv`` is intentionally not read because all
    rack-related constraints are removed in this relaxation.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    paths = {
        "orders.csv": data_dir / "orders.csv",
        "order_details.csv": data_dir / "order_details.csv",
        "products.csv": data_dir / "products.csv",
    }

    missing_files = [name for name, path in paths.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Data directory {data_dir} is missing: {', '.join(missing_files)}"
        )

    orders_df = pd.read_csv(paths["orders.csv"])
    details_df = pd.read_csv(paths["order_details.csv"])
    products_df = pd.read_csv(paths["products.csv"])

    _require_columns(orders_df, {"order_id", "weight"}, "orders.csv")
    _require_columns(
        details_df, {"order_id", "sku_id"}, "order_details.csv"
    )
    _require_columns(products_df, {"sku_id", "zone_id"}, "products.csv")

    if orders_df["order_id"].isna().any() or orders_df["weight"].isna().any():
        raise ValueError("orders.csv contains a missing order_id or weight.")
    if details_df[["order_id", "sku_id"]].isna().any().any():
        raise ValueError("order_details.csv contains a missing order_id or sku_id.")
    if products_df[["sku_id", "zone_id"]].isna().any().any():
        raise ValueError("products.csv contains a missing sku_id or zone_id.")
    if orders_df["order_id"].duplicated().any():
        duplicated = orders_df.loc[
            orders_df["order_id"].duplicated(keep=False), "order_id"
        ].tolist()
        raise ValueError(f"orders.csv contains duplicate order_id values: {duplicated}")

    weights = pd.to_numeric(orders_df["weight"], errors="coerce")
    if weights.isna().any():
        raise ValueError("orders.csv contains a non-numeric weight.")
    if (weights <= 0).any():
        invalid = orders_df.loc[weights <= 0, ["order_id", "weight"]]
        raise ValueError(
            "All order weights must be positive. Invalid rows: "
            f"{invalid.to_dict('records')}"
        )

    # A SKU must have exactly one zone assignment. Repeated identical rows are
    # harmless and are removed before building the mapping.
    sku_zone_pairs = products_df[["sku_id", "zone_id"]].drop_duplicates()
    conflicting_skus = (
        sku_zone_pairs.groupby("sku_id", dropna=False)["zone_id"].nunique()
    )
    conflicting_skus = conflicting_skus[conflicting_skus > 1]
    if not conflicting_skus.empty:
        raise ValueError(
            "Each SKU must belong to exactly one zone. Conflicting SKU(s): "
            f"{conflicting_skus.index.tolist()}"
        )

    order_info: Dict[OrderId, Dict[str, Any]] = {}
    for index in orders_df.index:
        oid = _python_scalar(orders_df.at[index, "order_id"])
        attributes = {
            column: _python_scalar(orders_df.at[index, column])
            for column in orders_df.columns
            if column != "order_id"
        }
        attributes["weight"] = float(weights.loc[index])
        order_info[oid] = attributes

    sku_to_zone: Dict[SkuId, ZoneId] = {
        _python_scalar(row["sku_id"]): _python_scalar(row["zone_id"])
        for _, row in sku_zone_pairs.iterrows()
    }
    zones = sorted(set(sku_to_zone.values()), key=_stable_id_key)

    order_needs: Dict[OrderId, Dict[ZoneId, Set[SkuId]]] = {
        oid: {zone: set() for zone in zones} for oid in order_info
    }

    unknown_orders: Set[OrderId] = set()
    unknown_skus: Set[SkuId] = set()
    for _, row in details_df.iterrows():
        oid = _python_scalar(row["order_id"])
        sku = _python_scalar(row["sku_id"])

        if oid not in order_info:
            unknown_orders.add(oid)
            continue
        if sku not in sku_to_zone:
            unknown_skus.add(sku)
            continue

        zone = sku_to_zone[sku]
        order_needs[oid][zone].add(sku)

    if unknown_orders:
        raise ValueError(
            "order_details.csv references order_id values absent from orders.csv: "
            f"{sorted(unknown_orders, key=_stable_id_key)}"
        )
    if unknown_skus:
        raise ValueError(
            "order_details.csv references sku_id values absent from products.csv: "
            f"{sorted(unknown_skus, key=_stable_id_key)}"
        )

    return order_info, order_needs, zones


def compute_lower_bound(
    order_info: Mapping[OrderId, Mapping[str, Any]],
    order_needs: Mapping[OrderId, Mapping[ZoneId, Set[SkuId]]],
    zones: Sequence[ZoneId],
    sigma: float = SIGMA,
    a: float = A,
    pi: float = PI,
) -> Dict[str, Any]:
    """Compute the global analytical lower bound from manuscript Section 3.4.

    Processing requirements are continuous: ``u_op = d_op / q``. In
    particular, this implementation must not replace ``u_op`` with
    ``ceil(d_op / q)``.
    """
    if sigma <= 0 or a <= 0:
        raise ValueError("sigma and a must both be positive.")
    if pi < 0:
        raise ValueError("pi must be nonnegative.")

    q = math.floor(sigma / a)
    if q < 1:
        raise ValueError(
            "floor(sigma / a) must be at least 1; increase sigma or reduce a."
        )

    normalized_weights: Dict[OrderId, float] = {}
    for oid, attributes in order_info.items():
        if "weight" not in attributes:
            raise ValueError(f"Order {oid!r} has no weight.")
        weight = float(attributes["weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Order {oid!r} has invalid weight {weight!r}.")
        normalized_weights[oid] = weight

    extra_orders = set(order_needs).difference(order_info)
    if extra_orders:
        raise ValueError(
            "order_needs contains orders absent from order_info: "
            f"{sorted(extra_orders, key=_stable_id_key)}"
        )

    total_weight = math.fsum(normalized_weights.values())
    baseline_bound = pi * total_weight

    zone_details: Dict[ZoneId, pd.DataFrame] = {}
    zone_rows: List[Dict[str, Any]] = []

    for zone in zones:
        jobs: List[Dict[str, Any]] = []

        for oid, weight in normalized_weights.items():
            demand = len(order_needs.get(oid, {}).get(zone, set()))
            if demand == 0:
                continue

            continuous_processing = demand / q
            jobs.append(
                {
                    "order_id": oid,
                    "zone_id": zone,
                    "weight": weight,
                    "d_op": demand,
                    "u_op": continuous_processing,
                    "wspt_ratio": continuous_processing / weight,
                }
            )

        if not jobs:
            continue

        jobs.sort(
            key=lambda job: (
                job["wspt_ratio"],
                _stable_id_key(job["order_id"]),
            )
        )

        cumulative_completion = 0.0
        weighted_completion_terms: List[float] = []
        for position, job in enumerate(jobs, start=1):
            cumulative_completion += job["u_op"]
            weighted_term = job["weight"] * cumulative_completion
            job["wspt_position"] = position
            job["continuous_completion"] = cumulative_completion
            job["weighted_continuous_completion"] = weighted_term
            weighted_completion_terms.append(weighted_term)

        zone_weight = math.fsum(job["weight"] for job in jobs)
        zone_objective = math.fsum(weighted_completion_terms)
        zone_increment = zone_objective - zone_weight
        zone_lower_bound = baseline_bound + sigma * zone_increment

        zone_details[zone] = pd.DataFrame(jobs)
        zone_rows.append(
            {
                "zone_id": zone,
                "number_of_orders": len(jobs),
                "W_p": zone_weight,
                "Z_p_cont": zone_objective,
                "Z_p_cont_minus_W_p": zone_increment,
                "LB_p": zone_lower_bound,
            }
        )

    strongest_zone_increment = max(
        (row["Z_p_cont_minus_W_p"] for row in zone_rows),
        default=float("-inf"),
    )
    selected_increment = max(0.0, strongest_zone_increment)
    global_lower_bound = baseline_bound + sigma * selected_increment

    if selected_increment == 0.0:
        binding_zones: List[ZoneId] = []
    else:
        binding_zones = [
            row["zone_id"]
            for row in zone_rows
            if math.isclose(
                row["Z_p_cont_minus_W_p"],
                selected_increment,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ]

    zone_summary = pd.DataFrame(
        zone_rows,
        columns=[
            "zone_id",
            "number_of_orders",
            "W_p",
            "Z_p_cont",
            "Z_p_cont_minus_W_p",
            "LB_p",
        ],
    )

    return {
        "lower_bound": global_lower_bound,
        "baseline_bound": baseline_bound,
        "selected_increment": selected_increment,
        "total_weight": total_weight,
        "zone_capacity_per_slot": q,
        "binding_zones": binding_zones,
        "zone_summary": zone_summary,
        "zone_details": zone_details,
        "parameters": {"pi": pi, "sigma": sigma, "a": a},
    }


def compute_gap_percent(objective_value: float, lower_bound: float) -> float:
    """Return ``(objective_value - lower_bound) / lower_bound * 100``."""
    if lower_bound <= 0:
        return float("nan")
    return (objective_value - lower_bound) / lower_bound * 100.0


def save_details(result: Mapping[str, Any], output_dir: Path | str) -> None:
    """Save the bound summary and WSPT sequences as date-stamped CSV files."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_row = {
        "lower_bound": result["lower_bound"],
        "baseline_bound": result["baseline_bound"],
        "selected_increment": result["selected_increment"],
        "total_weight": result["total_weight"],
        "q": result["zone_capacity_per_slot"],
        "binding_zones": ",".join(map(str, result["binding_zones"])),
        **result["parameters"],
    }
    pd.DataFrame([summary_row]).to_csv(
        output_dir / "lower_bound_summary_7_23.csv", index=False
    )
    result["zone_summary"].to_csv(
        output_dir / "lower_bound_by_zone_7_23.csv", index=False
    )

    for zone, details in result["zone_details"].items():
        safe_zone = str(zone).replace("/", "_").replace("\\", "_")
        details.to_csv(
            output_dir / f"lower_bound_zone_{safe_zone}_details_7_23.csv",
            index=False,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the analytical lower bound from manuscript Section 3.4 "
            "(updated 2026-07-23)."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory containing orders.csv, order_details.csv, and "
            "products.csv (default: current directory)."
        ),
    )
    parser.add_argument("--pi", type=float, default=PI)
    parser.add_argument("--sigma", type=float, default=SIGMA)
    parser.add_argument("--a", type=float, default=A)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for date-stamped CSV calculation details.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    order_info, order_needs, zones = load_lower_bound_data(args.data_dir)
    result = compute_lower_bound(
        order_info=order_info,
        order_needs=order_needs,
        zones=zones,
        sigma=args.sigma,
        a=args.a,
        pi=args.pi,
    )

    print("=" * 78)
    print("Section 3.4 analytical lower bound (updated 2026-07-23)")
    print("=" * 78)
    print(f"Data directory : {args.data_dir.expanduser().resolve()}")
    print(f"Orders        : {len(order_info)}")
    print(f"Zones         : {zones}")
    print(f"Parameters    : pi={args.pi:g}, sigma={args.sigma:g}, a={args.a:g}")
    print(f"q=floor(sigma/a): {result['zone_capacity_per_slot']}")
    print(f"W             : {result['total_weight']:.10f}")
    print(f"Baseline pi*W : {result['baseline_bound']:.10f}")
    print("-" * 78)

    if result["zone_summary"].empty:
        print("No non-empty zones; the baseline is the global lower bound.")
    else:
        print(
            result["zone_summary"].to_string(
                index=False,
                formatters={
                    "W_p": "{:.10f}".format,
                    "Z_p_cont": "{:.10f}".format,
                    "Z_p_cont_minus_W_p": "{:.10f}".format,
                    "LB_p": "{:.10f}".format,
                },
            )
        )

    binding_text = (
        ", ".join(map(str, result["binding_zones"]))
        if result["binding_zones"]
        else "baseline pi*W"
    )
    print("-" * 78)
    print(f"Binding source : {binding_text}")
    print(f"GLOBAL LB      : {result['lower_bound']:.10f}")

    if args.output_dir is not None:
        save_details(result, args.output_dir)
        print(f"Details saved  : {args.output_dir.expanduser().resolve()}")
    print("=" * 78)


if __name__ == "__main__":
    main()
