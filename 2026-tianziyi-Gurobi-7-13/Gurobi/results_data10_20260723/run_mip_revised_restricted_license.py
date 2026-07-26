"""Run mip_revised.py under Gurobi's size-limited license.

The original formulation adds six equivalent transitivity inequalities for
every unordered triple of orders.  Given x[i,j] + x[j,i] == 1, only the two
inequalities whose first order is the smallest member of the triple are
needed.  Removing the other four leaves the feasible region unchanged.
"""

from __future__ import annotations

import csv
import runpy
import sys
from pathlib import Path

import gurobipy as gp


def order_ids_from_current_case() -> list[int]:
    with Path("orders.csv").open(newline="", encoding="utf-8-sig") as handle:
        return [int(row["order_id"]) for row in csv.DictReader(handle)]


def install_equivalent_constraint_deduplication() -> None:
    original_optimize = gp.Model.optimize

    def optimize_with_deduplication(
        model: gp.Model, *args: object, **kwargs: object
    ) -> None:
        model.update()
        order_ids = order_ids_from_current_case()
        num_orders = len(order_ids)

        variables = model.getVars()
        num_gamma = sum(v.VarName.startswith("gamma[") for v in variables)
        num_y = sum(v.VarName.startswith("y[") for v in variables)
        num_z = sum(v.VarName.startswith("z[") for v in variables)

        num_slots = num_gamma // num_orders
        num_zones = num_z // num_slots
        num_order_skus = num_y // num_slots

        constraints_before = model.getConstrs()
        transitivity_start = (
            num_order_skus
            + num_slots * num_zones
            + num_y
            + num_slots * num_zones
            + num_orders * (num_orders - 1) // 2
        )
        num_transitivity = num_orders * (num_orders - 1) * (num_orders - 2)
        transitivity_end = transitivity_start + num_transitivity

        if transitivity_end > len(constraints_before):
            raise RuntimeError(
                "Could not locate the transitivity block in mip_revised.py; "
                "the original model layout may have changed."
            )

        ordered_triples = [
            (first, second, third)
            for first in order_ids
            for second in order_ids
            for third in order_ids
            if len({first, second, third}) == 3
        ]
        transitivity_constraints = constraints_before[
            transitivity_start:transitivity_end
        ]
        if len(transitivity_constraints) != len(ordered_triples):
            raise RuntimeError(
                "Transitivity constraint count does not match the order triples."
            )

        redundant = [
            constraint
            for constraint, triple in zip(
                transitivity_constraints, ordered_triples
            )
            if triple[0] != min(triple)
        ]
        model.remove(redundant)
        model.update()

        print(
            "\n>>> 受限许可证适配：删除数学等价的重复传递约束 "
            f"{len(redundant)} 条；当前变量 {model.NumVars} 个、"
            f"约束 {model.NumConstrs} 条 <<<\n"
        )
        original_optimize(model, *args, **kwargs)

    gp.Model.optimize = optimize_with_deduplication


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: run_mip_revised_restricted_license.py /path/to/mip_revised.py"
        )

    solver_path = Path(sys.argv[1]).resolve()
    install_equivalent_constraint_deduplication()
    runpy.run_path(str(solver_path), run_name="__main__")


if __name__ == "__main__":
    main()
