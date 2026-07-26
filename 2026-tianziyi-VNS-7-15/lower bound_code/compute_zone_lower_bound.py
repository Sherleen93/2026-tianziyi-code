import os
import math
from typing import Dict, Any, List, Set, Tuple
import time
import pandas as pd


# ================= 1. 参数设定：与大规模实验代码保持一致 =================
PI = 120       # completion offset: 若订单在第1个slot完成，则完成时间为120s
SIGMA = 20     # picking stage / slot interval
A = 10         # unit picking time for one medicine SKU
B = 10         # rack setup time, lower bound 中忽略，不参与计算
C = 10          # workbench capacity, lower bound 中松弛，不参与计算



# ================= 2. 数据加载：与原大规模实验代码保持一致 =================
def load_data():
    """
    Load the same dataset used in the large-scale VNS+DP comparison experiment.

    Required files:
        orders.csv
        order_details.csv
        products.csv
        inventory.csv

    Returns
    -------
    order_info:
        dict, order_id -> order attributes, including weight.

    order_needs:
        dict, order_id -> {zone_id -> set(sku_id)}

    rack_inv:
        dict, zone_id -> {rack_id -> set(sku_id)}

    zones:
        sorted list of zone ids

    order_total_demands:
        dict, order_id -> total number of required SKUs
    """

    try:
        orders_df = pd.read_csv("orders.csv")
        details_df = pd.read_csv("order_details.csv")
        products_df = pd.read_csv("products.csv")
        inventory_df = pd.read_csv("inventory.csv")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "未找到 orders.csv / order_details.csv / products.csv / inventory.csv。"
            "请将本代码放在这些CSV文件所在的文件夹下运行。"
        ) from e

    required_orders_cols = {"order_id", "weight"}
    required_details_cols = {"order_id", "sku_id"}
    required_products_cols = {"sku_id", "zone_id"}
    required_inventory_cols = {"rack_id", "sku_id", "zone_id"}

    if not required_orders_cols.issubset(set(orders_df.columns)):
        raise ValueError(f"orders.csv 至少需要包含列: {required_orders_cols}")

    if not required_details_cols.issubset(set(details_df.columns)):
        raise ValueError(f"order_details.csv 至少需要包含列: {required_details_cols}")

    if not required_products_cols.issubset(set(products_df.columns)):
        raise ValueError(f"products.csv 至少需要包含列: {required_products_cols}")

    if not required_inventory_cols.issubset(set(inventory_df.columns)):
        raise ValueError(f"inventory.csv 至少需要包含列: {required_inventory_cols}")

    order_info = orders_df.set_index("order_id").to_dict("index")

    sku_to_zone = products_df.set_index("sku_id")["zone_id"].to_dict()
    zones = sorted(list(set(sku_to_zone.values())))

    # order_needs[o][z] = set of SKUs required by order o in zone z
    order_needs: Dict[int, Dict[int, Set[int]]] = {}

    for _, row in details_df.iterrows():
        oid = int(row["order_id"])
        sku = int(row["sku_id"])

        z = sku_to_zone.get(sku)
        if z is None:
            # 与原代码一致：如果SKU没有zone信息，则跳过
            continue

        if oid not in order_needs:
            order_needs[oid] = {zn: set() for zn in zones}

        order_needs[oid][z].add(sku)

    # 若存在没有需求明细的订单，也补齐空需求，避免后续KeyError
    for oid in order_info.keys():
        if oid not in order_needs:
            order_needs[oid] = {zn: set() for zn in zones}

    # rack inventory
    rack_inv: Dict[int, Dict[int, Set[int]]] = {z: {} for z in zones}

    for _, row in inventory_df.iterrows():
        rid = int(row["rack_id"])
        sku = int(row["sku_id"])
        z = int(row["zone_id"])

        if z not in rack_inv:
            rack_inv[z] = {}

        if rid not in rack_inv[z]:
            rack_inv[z][rid] = set()

        rack_inv[z][rid].add(sku)

    order_total_demands = {
        oid: sum(len(skus) for skus in zone_dict.values())
        for oid, zone_dict in order_needs.items()
    }

    return order_info, order_needs, rack_inv, zones, order_total_demands


# ================= 3. Lower Bound 计算函数 =================
def compute_relaxed_final_completion_bound(
    order_info: Dict[int, Dict[str, Any]],
    order_needs: Dict[int, Dict[int, Set[int]]],
    zones: List[int],
    sigma: int = SIGMA,
    a: int = A,
    pi: int = PI,
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Compute relaxed order final completion times.

    Relaxations:
    1. Workbench capacity C is relaxed: all orders are active from the beginning.
    2. Rack revisits are ignored.
    3. Rack switching setup time is ignored.
    4. Rack sequencing is ignored.
    5. Each zone is treated as an idealized processor.
    6. For each zone, orders are sequenced by WSPT rule: p_{o,p} / w_o ascending.

    Important:
    This function calculates the final completion time of each order as:
        c_o = max_p c_{o,p}
        phi_o = sigma * (c_o - 1) + pi

    This gives a relaxed completion-time benchmark.
    """

    q = sigma // a

    if q <= 0:
        raise ValueError("Invalid parameters: SIGMA must be greater than or equal to A.")

    # 保存每个订单在每个 zone 的完成 slot
    order_zone_completion_slots: Dict[int, Dict[int, int]] = {
        oid: {} for oid in order_needs.keys()
    }

    zone_details: Dict[int, pd.DataFrame] = {}
    zone_bounds: Dict[int, float] = {}

    # =========================
    # Step 1: 每个 zone 单独排序并计算 c_{o,p}
    # =========================
    for z in zones:
        rows = []

        for oid, zone_dict in order_needs.items():
            d_op = len(zone_dict.get(z, set()))

            if d_op <= 0:
                continue

            if oid not in order_info:
                raise ValueError(f"Order {oid} appears in order_details.csv but not in orders.csv.")

            weight = float(order_info[oid]["weight"])

            if weight <= 0:
                raise ValueError(f"Order {oid} has non-positive weight: {weight}")

            # p_{o,p} = ceil(d_{o,p} / q)
            p_op = math.ceil(d_op / q)

            # WSPT ratio
            wspt_ratio = p_op / weight

            rows.append({
                "order_id": oid,
                "zone_id": z,
                "d_op": d_op,
                "p_op": p_op,
                "weight": weight,
                "wspt_ratio": wspt_ratio,
            })

        if not rows:
            zone_details[z] = pd.DataFrame()
            zone_bounds[z] = 0.0
            continue

        df_z = pd.DataFrame(rows)

        # WSPT: p_op / weight ascending
        df_z = df_z.sort_values(
            by=["wspt_ratio", "weight", "p_op", "order_id"],
            ascending=[True, False, True, True]
        ).reset_index(drop=True)

        cumulative_slot = 0
        completion_slots = []
        zone_completion_times = []
        zone_weighted_completion_times = []

        for _, row in df_z.iterrows():
            cumulative_slot += int(row["p_op"])

            oid = int(row["order_id"])
            c_op = cumulative_slot

            # 记录订单 o 在 zone p 的完成 slot
            order_zone_completion_slots[oid][z] = c_op

            # zone-level completion time
            phi_op = sigma * (c_op - 1) + pi

            completion_slots.append(c_op)
            zone_completion_times.append(phi_op)
            zone_weighted_completion_times.append(float(row["weight"]) * phi_op)

        df_z["zone_sequence_position"] = range(1, len(df_z) + 1)
        df_z["zone_completion_slot"] = completion_slots
        df_z["zone_completion_time"] = zone_completion_times
        df_z["zone_weighted_completion_time"] = zone_weighted_completion_times

        zone_details[z] = df_z
        zone_bounds[z] = float(df_z["zone_weighted_completion_time"].sum())

    # =========================
    # Step 2: 根据 max_p c_{o,p} 计算订单最终完成 slot 和完成时间
    # =========================
    final_rows = []

    for oid, zone_dict in order_needs.items():
        weight = float(order_info[oid]["weight"])

        related_zones = [
            z for z in zones
            if len(zone_dict.get(z, set())) > 0
        ]

        if not related_zones:
            # 如果某个订单没有任何需求，理论上可在slot 1完成
            final_completion_slot = 1
            zone_slot_dict = {}
        else:
            zone_slot_dict = {
                z: order_zone_completion_slots[oid][z]
                for z in related_zones
            }

            final_completion_slot = max(zone_slot_dict.values())

        # 你的完成时间公式
        final_completion_time = sigma * (final_completion_slot - 1) + pi

        weighted_completion_time = weight * final_completion_time

        final_rows.append({
            "order_id": oid,
            "weight": weight,
            "related_zones": ",".join(map(str, related_zones)),
            "zone_completion_slots": str(zone_slot_dict),
            "final_completion_slot": final_completion_slot,
            "final_completion_time": final_completion_time,
            "weighted_completion_time": weighted_completion_time,
        })

    final_order_details = pd.DataFrame(final_rows).sort_values(
        by=["final_completion_slot", "order_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    relaxed_objective_value = float(final_order_details["weighted_completion_time"].sum())

    result = {
        "relaxed_objective_value": relaxed_objective_value,
        "zone_capacity_per_slot": q,
        "zone_bounds": zone_bounds,
        "zone_details": zone_details,
        "final_order_details": final_order_details,
    }

    return result

def build_order_completion_explanation(
    relaxed_result: Dict[str, Any],
    order_info: Dict[int, Dict[str, Any]],
    order_needs: Dict[int, Dict[int, Set[int]]],
    zones: List[int],
    sigma: int = SIGMA,
    a: int = A,
    pi: int = PI,
) -> pd.DataFrame:
    """
    Build an order-level explanation table showing how each relaxed final
    completion time is obtained.

    For each order o, this table reports:
        d_{o,p}
        p_{o,p}
        WSPT ratio
        zone sequence position
        zone completion slot c_{o,p}
        final completion slot c_o = max_p c_{o,p}
        binding zone
        final completion time phi_o = sigma * (c_o - 1) + pi
        weighted completion time w_o * phi_o
    """

    q = sigma // a
    if q <= 0:
        raise ValueError("Invalid parameters: SIGMA must be greater than or equal to A.")

    # Convert zone_details into quick lookup:
    # lookup[(order_id, zone_id)] = row information
    zone_lookup = {}

    for z, df_z in relaxed_result["zone_details"].items():
        if df_z is None or df_z.empty:
            continue

        for _, row in df_z.iterrows():
            oid = int(row["order_id"])
            zone_lookup[(oid, z)] = {
                "d_op": int(row["d_op"]),
                "p_op": int(row["p_op"]),
                "weight": float(row["weight"]),
                "wspt_ratio": float(row["wspt_ratio"]),
                "zone_sequence_position": int(row["zone_sequence_position"]),
                "zone_completion_slot": int(row["zone_completion_slot"]),
                "zone_completion_time": float(row["zone_completion_time"]),
                "zone_weighted_completion_time": float(row["zone_weighted_completion_time"]),
            }

    explanation_rows = []

    for oid, zone_dict in order_needs.items():
        weight = float(order_info[oid]["weight"])

        related_zones = [
            z for z in zones
            if len(zone_dict.get(z, set())) > 0
        ]

        row_data = {
            "order_id": oid,
            "weight": weight,
            "related_zones": ",".join(map(str, related_zones)),
        }

        zone_completion_slots = {}

        for z in zones:
            d_op = len(zone_dict.get(z, set()))

            if d_op <= 0:
                row_data[f"zone_{z}_demand"] = 0
                row_data[f"zone_{z}_processing_slots"] = 0
                row_data[f"zone_{z}_wspt_ratio"] = ""
                row_data[f"zone_{z}_sequence_position"] = ""
                row_data[f"zone_{z}_completion_slot"] = ""
                row_data[f"zone_{z}_completion_time"] = ""
                continue

            info = zone_lookup[(oid, z)]

            row_data[f"zone_{z}_demand"] = info["d_op"]
            row_data[f"zone_{z}_processing_slots"] = info["p_op"]
            row_data[f"zone_{z}_wspt_ratio"] = info["wspt_ratio"]
            row_data[f"zone_{z}_sequence_position"] = info["zone_sequence_position"]
            row_data[f"zone_{z}_completion_slot"] = info["zone_completion_slot"]
            row_data[f"zone_{z}_completion_time"] = info["zone_completion_time"]

            zone_completion_slots[z] = info["zone_completion_slot"]

        if zone_completion_slots:
            final_completion_slot = max(zone_completion_slots.values())

            # 如果多个zone同时决定最终完成时间，全部记录
            binding_zones = [
                z for z, c in zone_completion_slots.items()
                if c == final_completion_slot
            ]

        else:
            # 无需求订单，理论上slot 1完成
            final_completion_slot = 1
            binding_zones = []

        final_completion_time = sigma * (final_completion_slot - 1) + pi
        weighted_completion_time = weight * final_completion_time

        row_data["zone_completion_slots_dict"] = str(zone_completion_slots)
        row_data["final_completion_slot"] = final_completion_slot
        row_data["binding_zone"] = ",".join(map(str, binding_zones))
        row_data["final_completion_time_formula"] = (
            f"{sigma} * ({final_completion_slot} - 1) + {pi}"
        )
        row_data["final_completion_time"] = final_completion_time
        row_data["weighted_completion_time_formula"] = (
            f"{weight} * {final_completion_time}"
        )
        row_data["weighted_completion_time"] = weighted_completion_time

        explanation_rows.append(row_data)

    explanation_df = pd.DataFrame(explanation_rows)

    explanation_df = explanation_df.sort_values(
        by=["final_completion_slot", "order_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    return explanation_df

def compute_gap_percent(ub: float, lb: float) -> float:
    """
    Gap = (UB - LB) / LB * 100%
    """
    if lb <= 0:
        return float("nan")
    return (ub - lb) / lb * 100.0


# ================= 6. 主程序 =================
if __name__ == "__main__":

    print("=" * 90)
    print("启动 relaxed final completion-time benchmark 计算")
    print("=" * 90)

    # ---------- Load data ----------
    ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS = load_data()

    print(f"订单数量: {len(ORDER_INFO)}")
    print(f"Zone集合: {ZONES}")
    print(f"参数: PI={PI}, SIGMA={SIGMA}, A={A}, B={B}, C={C}")

    print(
        "Relaxation: "
        "C relaxed, rack revisits ignored, "
        "setup time ignored, rack sequencing ignored."
    )

    print("-" * 90)

    # ---------- Runtime start ----------
    lb_start_time = time.perf_counter()

    # ---------- Compute relaxed benchmark ----------
    relaxed_result = compute_relaxed_final_completion_bound(
        order_info=ORDER_INFO,
        order_needs=ORDER_NEEDS_MASTER,
        zones=ZONES,
        sigma=SIGMA,
        a=A,
        pi=PI,
        return_details=False,
    )

    relaxed_value = relaxed_result["relaxed_objective_value"]

    # ---------- Runtime end ----------
    lb_end_time = time.perf_counter()

    lb_runtime_seconds = lb_end_time - lb_start_time

    # ---------- Print results ----------
    print("Zone-level auxiliary bounds:")

    for z, val in relaxed_result["zone_bounds"].items():
        print(f"  Zone {z}: {val:.4f}")

    print("-" * 90)

    print(
        f"每个zone每个slot的最大拣选能力 q = floor(SIGMA / A) = "
        f"{relaxed_result['zone_capacity_per_slot']}"
    )

    print(f"Relaxed final completion-time benchmark = {relaxed_value:.4f}")

    print(f"Runtime = {lb_runtime_seconds:.6f} seconds")

    print("=" * 90)
    print("Benchmark 计算完成")
    print("=" * 90)