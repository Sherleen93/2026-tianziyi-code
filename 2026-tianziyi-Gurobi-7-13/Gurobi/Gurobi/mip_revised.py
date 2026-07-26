import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import collections


# ==========================================
# 1. 数据预处理与参数设置
# ==========================================
def load_and_preprocess():
    orders_df = pd.read_csv("orders.csv")
    details_df = pd.read_csv("order_details.csv")
    inventory_df = pd.read_csv("inventory.csv")

    # D_o: 订单 o 所需药品集合
    D = collections.defaultdict(list)
    for _, row in details_df.iterrows():
        D[int(row["order_id"])].append(int(row["sku_id"]))

    # S_m: 存放药品 m 的货架集合
    S = collections.defaultdict(list)
    sku_to_p = {}
    for _, row in inventory_df.iterrows():
        m = int(row["sku_id"])
        r = int(row["rack_id"])
        p = int(row["zone_id"])
        if r not in S[m]:
            S[m].append(r)
        sku_to_p[m] = p

    rack_to_p = (
        inventory_df.drop_duplicates("rack_id")
        .set_index("rack_id")["zone_id"]
        .astype(int)
        .to_dict()
    )

    return orders_df, D, S, sku_to_p, rack_to_p


orders_df, D, S, sku_to_p, rack_to_p = load_and_preprocess()

# 参数
C = 3
a = 10
b = 10
sigma = 20
pi = 120

O = orders_df["order_id"].astype(int).tolist()
T = list(range(1, 15))
P = sorted(set(sku_to_p.values()))
R = sorted(rack_to_p.keys())

w = orders_df.set_index("order_id")["weight"].to_dict()

# Q_p: zone p 中的药品集合
Q = {
    p: {m for m, zone in sku_to_p.items() if zone == p}
    for p in P
}

# H_p: zone p 中的货架集合
H = {
    p: [r for r in R if rack_to_p[r] == p]
    for p in P
}


# ==========================================
# 2. Gurobi模型构建
# ==========================================
model = gp.Model("ORSP_New_Formulation")

# 变量
y = model.addVars(
    [(m, o, t) for o in O for m in D[o] for t in T],
    vtype=GRB.BINARY,
    name="y"
)

x = model.addVars(
    [(op, o) for op in O for o in O if op != o],
    vtype=GRB.BINARY,
    name="x"
)

gamma = model.addVars(O, T, vtype=GRB.BINARY, name="gamma")
z = model.addVars(T, P, vtype=GRB.BINARY, name="z")
alpha = model.addVars(R, T, vtype=GRB.BINARY, name="alpha")
phi = model.addVars(O, vtype=GRB.INTEGER, lb=0, name="phi")

# 目标函数（1）
model.setObjective(
    gp.quicksum(w[o] * phi[o] for o in O),
    GRB.MINIMIZE
)

# 约束（2）
for o in O:
    for m in D[o]:
        model.addConstr(
            gp.quicksum(y[m, o, t] for t in T) == 1
        )

# 约束（3）
for t in T:
    for p in P:
        model.addConstr(
            a * gp.quicksum(
                y[m, o, t]
                for o in O
                for m in D[o]
                if m in Q[p]
            )
            + b * z[t, p]
            <= sigma
        )

# 约束（4）
for o in O:
    for m in D[o]:
        for t in T:
            model.addConstr(
                y[m, o, t]
                <= gp.quicksum(alpha[r, t] for r in S[m])
            )

# 约束（5）
for t in T:
    for p in P:
        model.addConstr(
            gp.quicksum(alpha[r, t] for r in H[p]) == 1
        )

# 约束（6）
for o in O:
    for op in O:
        if o < op:
            model.addConstr(
                x[op, o] + x[o, op] == 1
            )

# 约束（7）
for op in O:
    for o in O:
        for opp in O:
            if len({op, o, opp}) == 3:
                model.addConstr(
                    x[op, o] + x[o, opp] - 1 <= x[op, opp]
                )

# 约束（8）
# 其中 gamma[o,0] = 0(约束条件12) 作为初始设定
for o in O:
    order_position = 1 + gp.quicksum(
        x[op, o] for op in O if op != o
    )

    for m in D[o]:
        for t in T:
            completed_before = (
                gp.quicksum(gamma[j, t - 1] for j in O)
                if t >= 2 else 0
            )

            model.addConstr(
                order_position
                <= C
                + completed_before
                + len(O) * (1 - y[m, o, t])
            )

# 约束（9）和（10）
for o in O:
    num_m = len(D[o])

    for t in T:
        picked_so_far = gp.quicksum(
            y[m, o, tau]
            for tau in range(1, t + 1)
            for m in D[o]
        )

        model.addConstr(
            picked_so_far >= num_m * gamma[o, t]
        )

        model.addConstr(
            picked_so_far <= num_m - 1 + gamma[o, t]
        )

# 约束（11）
for o in O:
    for t in range(2, len(T) + 1):
        model.addConstr(
            gamma[o, t] >= gamma[o, t - 1]
        )

# 约束（13）和（14）
for p in P:
    for r in H[p]:
        for t in range(2, len(T) + 1):
            model.addConstr(
                alpha[r, t] - alpha[r, t - 1] <= z[t, p]
            )

            model.addConstr(
                2 - alpha[r, t] - alpha[r, t - 1] >= z[t, p]
            )

# 约束（15）
for p in P:
    for r in H[p]:
        model.addConstr(
            alpha[r, 1] <= z[1, p]
        )

# 约束（16）
for o in O:
    model.addConstr(
        phi[o]
        == sigma * gp.quicksum(1 - gamma[o, t] for t in T)
        + pi
    )


print("\n>>> 正在运行：新模型详细输出版 V2 <<<\n")

# ==========================================
# 3. 求解
# ==========================================
model.Params.TimeLimit = 1800
model.optimize()


# ==========================================
# 4. 结果展示
# ==========================================
if model.SolCount > 0:
    print("\n" + "=" * 60)
    print("【第一部分】订单整体执行看板")
    print("=" * 60)

    # 订单在激活序列中的位置
    sequence_position = {
        o: 1 + sum(
            1
            for op in O
            if op != o and x[op, o].X > 0.5
        )
        for o in O
    }
    sequence = sorted(O, key=lambda o: sequence_position[o])

    # 完成slot
    completion_slot = {
        o: min(t for t in T if gamma[o, t].X > 0.5)
        for o in O
    }

    # 根据排序位置和已完成订单数推导激活slot
    activation_slot = {}
    for o in O:
        for t in T:
            completed_before = (
                sum(gamma[j, t - 1].X > 0.5 for j in O)
                if t >= 2 else 0
            )
            if sequence_position[o] <= C + completed_before:
                activation_slot[o] = t
                break

    print("订单激活顺序:", " -> ".join(map(str, sequence)))
    print(f"最优目标值: {model.ObjVal:.1f}\n")

    for o in sequence:
        print(
            f"订单{o}: "
            f"序列位置={sequence_position[o]}, "
            f"激活Slot={activation_slot[o]}, "
            f"完成Slot={completion_slot[o]}, "
            f"完成时间={phi[o].X:.0f}s"
        )

    print("\n" + "=" * 60)
    print("【第二部分】各 Slot 精细作业流水账")
    print("=" * 60)

    p_names = {
        1: "常温库区(P1)",
        2: "阴凉库区(P2)",
        3: "冷冻库区(P3)"
    }

    last_slot = max(completion_slot.values())

    for t in range(1, last_slot + 1):
        print(f"\n▶▶ Slot {t} (时间: {(t - 1) * sigma}s - {t * sigma}s)")

        # 1. 激活订单
        activated_orders = [
            o for o in sequence if activation_slot[o] == t
        ]
        if activated_orders:
            print(
                f"  [投产指令] 分配至工作台开始作业的订单: "
                f"{activated_orders}"
            )

        # 2. 各库区货架与拣选明细
        for p in P:
            active_rack = next(
                (r for r in H[p] if alpha[r, t].X > 0.5),
                None
            )

            picks = [
                f"订单{o}(需药品{m})"
                for o in O
                for m in D[o]
                if m in Q[p] and y[m, o, t].X > 0.5
            ]

            rack_str = f"R{active_rack}" if active_rack is not None else "空闲"
            picks_str = "，".join(picks) if picks else "无拣选作业"
            switch_str = "是" if z[t, p].X > 0.5 else "否"

            print(
                f"  [{p_names.get(p, f'库区(P{p})')}] "
                f"=> 分配停靠货架: {rack_str} | "
                f"货架切换: {switch_str} | "
                f"执行动作: {picks_str}"
            )

        # 3. 本slot完成订单
        completed_orders = [
            o for o in O
            if gamma[o, t].X > 0.5
            and (t == 1 or gamma[o, t - 1].X < 0.5)
        ]
        if completed_orders:
            print(
                f"  [流转指令] 订单 {completed_orders} 完成，"
                f"释放工作台位置"
            )

else:
    print(f"求解失败，无可行解。Status Code: {model.Status}")