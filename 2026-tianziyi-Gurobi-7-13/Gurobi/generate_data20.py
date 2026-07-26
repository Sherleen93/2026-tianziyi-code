"""Generate reproducible 20-order supplemental instances.

Each instance is a priority-stratified subset of its corresponding 30-order
benchmark.  The product catalog, rack inventory, zone assignment, beta, and
delta setting are retained.  Six emergency and fourteen normal orders are
sampled so the emergency-order proportion remains exactly alpha = 0.3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Iterable


CASE_NAMES = [
    "Medical_mu1_beta6_alpha03_rack12",
    "Medical_mu1_beta6_alpha03_rack15",
    "Medical_mu1_beta8_alpha03_rack12",
    "Medical_mu1_beta8_alpha03_rack15",
    "Medical_mu2_beta6_alpha03_rack12",
    "Medical_mu2_beta6_alpha03_rack15",
    "Medical_mu2_beta8_alpha03_rack12",
    "Medical_mu2_beta8_alpha03_rack15",
]

BASE_SAMPLING_SEED = 20260723
NUM_ORDERS = 20
NUM_EMERGENCY = 6
NUM_NORMAL = NUM_ORDERS - NUM_EMERGENCY


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(
    path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def sampling_seed(case_name: str) -> int:
    digest = hashlib.sha256(case_name.encode("utf-8")).digest()
    return BASE_SAMPLING_SEED + int.from_bytes(digest[:4], "big") % 1_000_000


def source_directory(source_root: Path, case_name: str) -> Path:
    direct = source_root / case_name
    if direct.exists():
        return direct

    if case_name == "Medical_mu2_beta8_alpha03_rack15":
        c3_source = source_root / f"{case_name}_C3"
        if c3_source.exists():
            return c3_source

    raise FileNotFoundError(f"No 30-order source dataset found for {case_name}")


def is_emergency(order: dict[str, str]) -> bool:
    return (
        order.get("type", "").casefold() == "emergency"
        or order.get("priority", "").casefold() == "high"
        or float(order["weight"]) > 1.0
    )


def generate_case(
    source_root: Path, output_root: Path, case_name: str
) -> dict[str, object]:
    source_dir = source_directory(source_root, case_name)
    output_dir = output_root / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    products = read_rows(source_dir / "products.csv")
    inventory = read_rows(source_dir / "inventory.csv")
    orders = read_rows(source_dir / "orders.csv")
    details = read_rows(source_dir / "order_details.csv")
    source_summary = read_rows(source_dir / "dataset_summary.csv")[0]

    emergency_orders = [row for row in orders if is_emergency(row)]
    normal_orders = [row for row in orders if not is_emergency(row)]
    if len(emergency_orders) < NUM_EMERGENCY or len(normal_orders) < NUM_NORMAL:
        raise ValueError(f"{case_name} does not contain enough orders by priority")

    case_seed = sampling_seed(case_name)
    rng = random.Random(case_seed)
    selected = rng.sample(emergency_orders, NUM_EMERGENCY)
    selected += rng.sample(normal_orders, NUM_NORMAL)
    selected.sort(key=lambda row: int(row["order_id"]))

    old_to_new = {
        int(order["order_id"]): new_id
        for new_id, order in enumerate(selected, start=1)
    }
    selected_old_ids = list(old_to_new)

    generated_orders: list[dict[str, object]] = []
    for order in selected:
        generated_order = dict(order)
        generated_order["order_id"] = old_to_new[int(order["order_id"])]
        generated_orders.append(generated_order)

    generated_details: list[dict[str, object]] = []
    for detail in details:
        old_id = int(detail["order_id"])
        if old_id not in old_to_new:
            continue
        generated_detail = dict(detail)
        generated_detail["order_id"] = old_to_new[old_id]
        generated_details.append(generated_detail)

    order_sizes: dict[int, int] = {}
    for detail in generated_details:
        order_id = int(detail["order_id"])
        order_sizes[order_id] = order_sizes.get(order_id, 0) + 1

    inventory_skus = {int(row["sku_id"]) for row in inventory}
    requested_skus = {int(row["sku_id"]) for row in generated_details}
    racks = {int(row["rack_id"]) for row in inventory}

    if len(generated_orders) != NUM_ORDERS:
        raise ValueError(f"{case_name}: expected {NUM_ORDERS} generated orders")
    if sum(is_emergency(row) for row in generated_orders) != NUM_EMERGENCY:
        raise ValueError(f"{case_name}: emergency ratio is not 0.3")
    if set(order_sizes) != set(range(1, NUM_ORDERS + 1)):
        raise ValueError(f"{case_name}: at least one order has no SKU details")
    if any(size < 1 or size > 5 for size in order_sizes.values()):
        raise ValueError(f"{case_name}: order size is outside [1, 5]")
    if not requested_skus.issubset(inventory_skus):
        raise ValueError(f"{case_name}: an ordered SKU is absent from inventory")

    write_rows(output_dir / "products.csv", products[0].keys(), products)
    write_rows(output_dir / "inventory.csv", inventory[0].keys(), inventory)
    write_rows(output_dir / "orders.csv", orders[0].keys(), generated_orders)
    write_rows(
        output_dir / "order_details.csv", details[0].keys(), generated_details
    )

    summary = dict(source_summary)
    source_seed = summary["seed"]
    summary.update(
        {
            "exp_name": case_name,
            "seed": case_seed,
            "num_orders": NUM_ORDERS,
            "num_skus": len(products),
            "num_racks": len(racks),
            "source_dataset": source_dir.name,
            "source_seed": source_seed,
            "sampling_seed": case_seed,
            "generation_method": "priority_stratified_subset_of_30_orders",
        }
    )
    write_rows(output_dir / "dataset_summary.csv", summary.keys(), [summary])

    return {
        "case": case_name,
        "source_dataset": source_dir.name,
        "sampling_seed": case_seed,
        "selected_source_order_ids": " ".join(map(str, selected_old_ids)),
        "num_orders": len(generated_orders),
        "num_emergency": NUM_EMERGENCY,
        "num_normal": NUM_NORMAL,
        "num_order_details": len(generated_details),
        "mean_order_size": f"{len(generated_details) / NUM_ORDERS:.3f}",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parent / "数据" / "30",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "数据" / "20",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        generate_case(args.source_root, args.output_root, case_name)
        for case_name in CASE_NAMES
    ]
    write_rows(
        args.output_root / "data20_generation_manifest.csv",
        manifest[0].keys(),
        manifest,
    )

    for row in manifest:
        print(
            f"{row['case']}: orders={row['num_orders']}, "
            f"emergency={row['num_emergency']}, "
            f"details={row['num_order_details']}, "
            f"seed={row['sampling_seed']}"
        )


if __name__ == "__main__":
    main()
