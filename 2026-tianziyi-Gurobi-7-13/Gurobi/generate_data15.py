"""Generate reproducible 15-order supplemental instances.

The instances are priority-stratified subsets of the corresponding 30-order
benchmarks.  Because 15 * 0.3 = 4.5, each instance uses five emergency orders
and ten normal orders (actual emergency ratio 1/3).  Both the target and actual
ratios are recorded in dataset_summary.csv.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from generate_data20 import (
    CASE_NAMES,
    is_emergency,
    read_rows,
    source_directory,
    write_rows,
)


BASE_SAMPLING_SEED = 20260723015
NUM_ORDERS = 15
NUM_EMERGENCY = 5
NUM_NORMAL = NUM_ORDERS - NUM_EMERGENCY
TARGET_ALPHA = 0.3
ACTUAL_ALPHA = NUM_EMERGENCY / NUM_ORDERS


def sampling_seed(case_name: str) -> int:
    digest = hashlib.sha256(f"data15:{case_name}".encode("utf-8")).digest()
    return BASE_SAMPLING_SEED + int.from_bytes(digest[:4], "big") % 1_000_000


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
        raise ValueError(f"{case_name}: expected {NUM_EMERGENCY} emergency orders")
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
            "alpha_emergency_ratio": f"{ACTUAL_ALPHA:.12f}",
            "target_alpha_emergency_ratio": TARGET_ALPHA,
            "actual_emergency_ratio": f"{ACTUAL_ALPHA:.12f}",
            "num_emergency_orders": NUM_EMERGENCY,
            "num_normal_orders": NUM_NORMAL,
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
        "target_alpha": TARGET_ALPHA,
        "actual_alpha": f"{ACTUAL_ALPHA:.12f}",
        "num_order_details": len(generated_details),
        "mean_order_size": f"{len(generated_details) / NUM_ORDERS:.3f}",
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    source_root = root / "数据" / "30"
    output_root = root / "数据" / "15"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = [
        generate_case(source_root, output_root, case_name)
        for case_name in CASE_NAMES
    ]
    write_rows(
        output_root / "data15_generation_manifest.csv",
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
