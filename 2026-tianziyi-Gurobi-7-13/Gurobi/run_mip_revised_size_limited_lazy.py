"""Execute mip_revised.py exactly with selected constraints enforced lazily.

The local Gurobi license permits at most 2,000 variables and 2,000 explicit
linear constraints.  Twenty-order instances fit the variable limit but the
original formulation exceeds the constraint limit.  This runner leaves the
original model-building file unchanged and removes four constraint blocks
before optimization:

* all explicitly enumerated order-transitivity constraints;
* order-activation constraints;
* the upper completion-indicator inequalities; and
* completion-indicator monotonicity constraints.

Every removed inequality is checked at each integer candidate and reintroduced
through cbLazy when violated.  Therefore, every accepted solution satisfies the
same constraints as the original formulation.
"""

from __future__ import annotations

import csv
import itertools
import os
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import gurobipy as gp
from gurobipy import GRB


def read_order_skus() -> tuple[list[int], dict[int, list[int]]]:
    with Path("orders.csv").open(newline="", encoding="utf-8-sig") as handle:
        order_ids = [
            int(row["order_id"])
            for row in csv.DictReader(handle)
        ]

    order_skus: dict[int, list[int]] = defaultdict(list)
    with Path("order_details.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            order_skus[int(row["order_id"])].append(int(row["sku_id"]))

    return order_ids, dict(order_skus)


def integer_parameter(script_path: Path, name: str) -> int:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(\d+)\s*$")
    for line in script_path.read_text(encoding="utf-8-sig").splitlines():
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not read integer parameter {name} from {script_path}")


def tuple_indices(var_name: str, prefix: str) -> tuple[int, ...] | None:
    match = re.fullmatch(rf"{re.escape(prefix)}\[(.+)\]", var_name)
    if not match:
        return None
    return tuple(int(part) for part in match.group(1).split(","))


def install_size_limited_lazy_adapter(script_path: Path) -> None:
    original_optimize = gp.Model.optimize
    workbench_capacity = integer_parameter(script_path, "C")

    def optimize_with_lazy_constraints(
        model: gp.Model, *args: object, **kwargs: object
    ) -> None:
        if args or kwargs:
            raise TypeError(
                "The size-limited runner expects mip_revised.py to call "
                "model.optimize() without an existing callback."
            )

        model.update()
        order_ids, order_skus = read_order_skus()
        order_ids = sorted(order_ids)
        num_orders = len(order_ids)

        x: dict[tuple[int, int], gp.Var] = {}
        y: dict[tuple[int, int, int], gp.Var] = {}
        gamma: dict[tuple[int, int], gp.Var] = {}
        for variable in model.getVars():
            if (indices := tuple_indices(variable.VarName, "x")) is not None:
                x[indices] = variable
            elif (indices := tuple_indices(variable.VarName, "y")) is not None:
                y[indices] = variable
            elif (
                indices := tuple_indices(variable.VarName, "gamma")
            ) is not None:
                gamma[indices] = variable

        slots = sorted({time for _, time in gamma})
        num_slots = len(slots)
        num_y = len(y)
        num_order_skus = num_y // num_slots
        num_zones = sum(
            variable.VarName.startswith("z[")
            for variable in model.getVars()
        ) // num_slots

        constraints = model.getConstrs()
        cursor = 0
        cursor += num_order_skus
        cursor += num_slots * num_zones
        cursor += num_y
        cursor += num_slots * num_zones
        cursor += num_orders * (num_orders - 1) // 2

        num_transitivity = num_orders * (num_orders - 1) * (num_orders - 2)
        transitivity_constraints = constraints[
            cursor:cursor + num_transitivity
        ]
        cursor += num_transitivity

        activation_constraints = constraints[cursor:cursor + num_y]
        cursor += num_y

        completion_pair_count = 2 * num_orders * num_slots
        completion_pairs = constraints[cursor:cursor + completion_pair_count]
        completion_upper_constraints = completion_pairs[1::2]
        cursor += completion_pair_count

        monotonicity_count = num_orders * (num_slots - 1)
        monotonicity_constraints = constraints[
            cursor:cursor + monotonicity_count
        ]

        expected_counts = (
            len(transitivity_constraints) == num_transitivity,
            len(activation_constraints) == num_y,
            len(completion_upper_constraints) == num_orders * num_slots,
            len(monotonicity_constraints) == monotonicity_count,
        )
        if not all(expected_counts):
            raise RuntimeError(
                "Could not locate the expected constraint blocks; "
                "mip_revised.py may have changed."
            )

        removed_constraints = [
            *transitivity_constraints,
            *activation_constraints,
            *completion_upper_constraints,
            *monotonicity_constraints,
        ]
        model.remove(removed_constraints)
        model.update()

        if model.NumVars > 2000 or model.NumConstrs > 2000:
            raise RuntimeError(
                "Adapted model still exceeds the size-limited license: "
                f"{model.NumVars} variables, {model.NumConstrs} constraints."
            )

        relevant_variables = list(x.values()) + list(y.values()) + list(
            gamma.values()
        )
        lazy_counts = {
            "transitivity": 0,
            "activation": 0,
            "completion_upper": 0,
            "monotonicity": 0,
        }
        callback_errors: list[BaseException] = []

        def lazy_callback(callback_model: gp.Model, where: int) -> None:
            if where != GRB.Callback.MIPSOL:
                return

            try:
                solution_values = callback_model.cbGetSolution(
                    relevant_variables
                )
                value = dict(zip(relevant_variables, solution_values))
                tolerance = 1e-6

                for first, second, third in itertools.combinations(
                    order_ids, 3
                ):
                    expression = (
                        x[first, second]
                        + x[second, third]
                        - x[first, third]
                    )
                    expression_value = (
                        value[x[first, second]]
                        + value[x[second, third]]
                        - value[x[first, third]]
                    )
                    if expression_value > 1.0 + tolerance:
                        callback_model.cbLazy(expression <= 1)
                        lazy_counts["transitivity"] += 1
                    elif expression_value < -tolerance:
                        callback_model.cbLazy(expression >= 0)
                        lazy_counts["transitivity"] += 1

                order_position_value = {
                    order: 1.0
                    + sum(
                        value[x[other, order]]
                        for other in order_ids
                        if other != order
                    )
                    for order in order_ids
                }
                completed_before_value = {
                    time: (
                        sum(
                            value[gamma[order, time - 1]]
                            for order in order_ids
                        )
                        if time >= 2
                        else 0.0
                    )
                    for time in slots
                }

                for (sku, order, time), y_variable in y.items():
                    if value[y_variable] <= 0.5:
                        continue
                    if (
                        order_position_value[order]
                        <= workbench_capacity
                        + completed_before_value[time]
                        + tolerance
                    ):
                        continue

                    order_position = 1 + gp.quicksum(
                        x[other, order]
                        for other in order_ids
                        if other != order
                    )
                    completed_before = (
                        gp.quicksum(
                            gamma[other, time - 1]
                            for other in order_ids
                        )
                        if time >= 2
                        else 0
                    )
                    callback_model.cbLazy(
                        order_position
                        <= workbench_capacity
                        + completed_before
                        + num_orders * (1 - y_variable)
                    )
                    lazy_counts["activation"] += 1

                for order in order_ids:
                    num_skus = len(order_skus[order])
                    picked_so_far_value = 0.0
                    for time in slots:
                        picked_so_far_value += sum(
                            value[y[sku, order, time]]
                            for sku in order_skus[order]
                        )
                        gamma_variable = gamma[order, time]
                        if (
                            picked_so_far_value
                            <= num_skus
                            - 1
                            + value[gamma_variable]
                            + tolerance
                        ):
                            continue

                        picked_so_far = gp.quicksum(
                            y[sku, order, tau]
                            for tau in slots
                            if tau <= time
                            for sku in order_skus[order]
                        )
                        callback_model.cbLazy(
                            picked_so_far
                            <= num_skus - 1 + gamma_variable
                        )
                        lazy_counts["completion_upper"] += 1

                    for time in slots[1:]:
                        if (
                            value[gamma[order, time]]
                            + tolerance
                            >= value[gamma[order, time - 1]]
                        ):
                            continue
                        callback_model.cbLazy(
                            gamma[order, time]
                            >= gamma[order, time - 1]
                        )
                        lazy_counts["monotonicity"] += 1
            except BaseException as error:
                callback_errors.append(error)
                callback_model.terminate()

        model.Params.LazyConstraints = 1
        if thread_limit := os.environ.get("GUROBI_THREADS"):
            model.Params.Threads = int(thread_limit)
        print(
            "\n>>> 受限许可证精确适配："
            f"移除 {len(removed_constraints)} 条显式约束并改为延迟约束；"
            f"初始模型 {model.NumVars} 个变量、"
            f"{model.NumConstrs} 条约束 <<<\n"
        )
        original_optimize(model, lazy_callback)

        if callback_errors:
            raise RuntimeError("Lazy-constraint callback failed") from (
                callback_errors[0]
            )

        print(
            "\n>>> 延迟约束加入统计："
            + ", ".join(
                f"{name}={count}" for name, count in lazy_counts.items()
            )
            + " <<<\n"
        )

    gp.Model.optimize = optimize_with_lazy_constraints


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: run_mip_revised_size_limited_lazy.py "
            "/path/to/mip_revised.py"
        )

    script_path = Path(sys.argv[1]).resolve()
    install_size_limited_lazy_adapter(script_path)
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()
