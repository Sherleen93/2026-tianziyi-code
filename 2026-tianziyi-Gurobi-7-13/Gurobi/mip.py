import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import collections
import math


# ==========================================
# 1. 数据预处理与参数设置
# ==========================================
def load_and_preprocess():
    # 假定本地已经存在了这3个csv文件，且 orders.csv 包含 weight 和 due_time
    orders_df = pd.read_csv('orders.csv')
    details_df = pd.read_csv('order_details.csv')
    inventory_df = pd.read_csv('inventory.csv')

    # 订单需求映射 D[i] = [m1, m2, ...]
    D = collections.defaultdict(list)
    for _, r in details_df.iterrows():
        D[int(r['order_id'])].append(int(r['sku_id']))

    # 货架与工位映射
    S = collections.defaultdict(list)
    sku_to_p = {}  # 药品到唯一工位的映射
    for _, r in inventory_df.iterrows():
        sku_id, rack_id, p_idx = int(r['sku_id']), int(r['rack_id']), int(r['zone_id'])
        S[sku_id].append(rack_id)
        sku_to_p[sku_id] = p_idx

    rack_to_p = inventory_df.set_index('rack_id')['zone_id'].to_dict()
    rack_to_p = {k: v  for k, v in rack_to_p.items()}

    return orders_df, D, S, sku_to_p, rack_to_p


orders_df, D, S, sku_to_p, rack_to_p = load_and_preprocess()

# --- 物理与时间参数 ---
sigma = 20  # 每个 拣货 的时间阈值
pi = 120  # 每个slot的总完成时间
a = 10  # 每种药品的拣货时间
b = 10  # 货架切换时间
C = 3  # 打包台最大并发容量

O = orders_df['order_id'].tolist()
T = list(range(1, 15))  # 时间槽集合，设为20以保证充足调度空间
P_stations = [1, 2, 3]
R = list(rack_to_p.keys())

# 权重与截单时间
wi = orders_df.set_index('order_id')['weight'].to_dict()



# ==========================================
# 2. Gurobi 模型构建
# ==========================================
model = gp.Model("Optimized_Pharmacy_Min_Tardy_Orders")

# --- 变量定义 ---
# (21) 布尔变量
y = model.addVars([(m, i, t, sku_to_p[m]) for i in O for m in D[i] for t in T], vtype=GRB.BINARY, name="y")
alpha = model.addVars([(r, t, rack_to_p[r]) for r in R for t in T], vtype=GRB.BINARY, name="alpha")
x = model.addVars(O, T, vtype=GRB.BINARY, name="x")
beta = model.addVars(O, T, vtype=GRB.BINARY, name="beta")
gamma = model.addVars(O, T, vtype=GRB.BINARY, name="gamma")
z = model.addVars(T, P_stations, vtype=GRB.BINARY, name="z")
mu = model.addVars(O, vtype=GRB.INTEGER, name="mu")

model.setObjective(gp.quicksum(wi[i] * mu[i] for i in O), GRB.MINIMIZE)

# --- 核心约束 ---
#约束（2）、（6）、（8）
for i in O:
    num_m = len(D[i])
    for m in D[i]:
        p_m = sku_to_p[m]
        model.addConstr(gp.quicksum(y[m, i, t, p_m] for t in T) == 1)

    for t in T:
        y_sum_it = gp.quicksum(y[m, i, t, sku_to_p[m]] for m in D[i])
        model.addConstr(num_m * x[i, t] >= y_sum_it)
        model.addConstr(x[i, t] <= y_sum_it)

    # 约束（7）、（9）、（10）、（11）、（12）
   # for t in range(2, len(T) + 1):
       # picked_so_far = gp.quicksum(y[m, i, tau, sku_to_p[m]] for tau in range(1, t) for m in D[i])
        #model.addConstr(x[i, t] >= x[i, t - 1] - (picked_so_far / num_m))

    for t in T:
        y_sum_it = gp.quicksum(y[m, i, t, sku_to_p[m]] for m in D[i])
        beta_cum = gp.quicksum(beta[i, tau] for tau in range(1, t + 1))
        model.addConstr(y_sum_it <= num_m * beta_cum)

        y_cum = gp.quicksum(y[m, i, tau, sku_to_p[m]] for tau in range(1, t + 1) for m in D[i])
        model.addConstr(y_cum >= num_m * gamma[i, t])
        model.addConstr(gamma[i, t] <= beta_cum)

    for t in range(2, len(T) + 1):
        model.addConstr(gamma[i, t] >= gamma[i, t - 1])

    # 约束（16）、（13）、（18）、（19）
    model.addConstr(beta.sum(i, '*') == 1)
    model.addConstr(gamma.sum(i, '*') >= 1)
    model.addConstr(mu[i] == sigma * gp.quicksum(1 - gamma[i, t] for t in T) + pi)

# 约束（14）、（15）
for t in T:
    occupied = gp.quicksum(
        gp.quicksum(beta[i, tau] for tau in range(1, t + 1)) - (gamma[i, t - 1] if t > 1 else 0) for i in O)
    model.addConstr(occupied <= C)

# 约束（3）、（4）、（5）、（17）
for t in T:
    for p in P_stations:
        y_at_p = [y[m, i, t, p] for i in O for m in D[i] if sku_to_p[m] == p]
        model.addConstr(gp.quicksum(a * v for v in y_at_p) + b * z[t, p] <= sigma)

        r_at_p = [r for r in R if rack_to_p[r] == p]
        model.addConstr(gp.quicksum(alpha[r, t, p] for r in r_at_p) <= 1)

        for r in r_at_p:
            if t == 1:
                model.addConstr(alpha[r, 1, p] <= z[1, p])
            else:
                model.addConstr(alpha[r, t, p] - alpha[r, t - 1, p] <= z[t, p])

for (m, i, t, p) in y.keys():
    r_matching = S[m]
    model.addConstr(y[m, i, t, p] <= gp.quicksum(alpha[r, t, p] for r in r_matching))

# ==========================================
# 3. 求解参数调优 (加速版)
# ==========================================
model.Params.TimeLimit = 1800  # 30分钟限制
model.optimize()

# ==========================================
# 4. 结果展示与作业流水日志
# ==========================================
if model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and model.SolCount > 0:

    print("\n" + "=" * 60)
    print("【第一部分】订单整体执行看板")
    print("=" * 60)
    res_list = []
    for i in O:
        start_t = [t for t in T if beta[i, t].X > 0.5][0]
        end_t = int(round(mu[i].X))
        actual_time = sigma * (end_t - 1) + pi
        res_list.append({
            "订单号": i, "权重": wi[i], "投产Slot": start_t, "完工Slot": end_t
        })
    df_res = pd.DataFrame(res_list).sort_values(by="投产Slot")
    print(df_res.to_string(index=False))
    print(f"\n=> 最终优化目标得分: {model.ObjVal:.1f}")

    print("\n" + "=" * 60)
    print("【第二部分】各 Slot 精细作业流水账")
    print("=" * 60)

    for t in T:
        slot_has_action = False
        log_lines = []

        # 1. 投产判定
        started_orders = [i for i in O if beta[i, t].X > 0.5]
        if started_orders:
            log_lines.append(f"  [投产指令] 分配至打包台开始作业的订单: {started_orders}")
            slot_has_action = True

        # 2. 货架与拣选明细
        p_names = {1: "常温库区(P1)", 2: "阴凉库区(P2)", 3: "冷冻库区(P3)"}
        for p in P_stations:
            # 确定停靠货架
            active_rack = None
            for r in [r for r in R if rack_to_p[r] == p]:
                if alpha[r, t, p].X > 0.5:
                    active_rack = r
                    break

            # 确定拣选药品
            picks = []
            for (m, i, t_y, p_y) in y.keys():
                if t_y == t and p_y == p:
                    if y[m, i, t, p].X > 0.5:
                        picks.append(f"订单{i}(需药品{m})")

            if active_rack or picks:
                rack_str = f"R{active_rack}" if active_rack else "空闲"
                picks_str = ", ".join(picks) if picks else "无拣选作业"
                log_lines.append(f"  [{p_names[p]}] => 分配停靠货架: {rack_str} | 执行动作: {picks_str}")
                slot_has_action = True

        # 3. 完工释放判定
        completed_orders = []
        for i in O:
            if gamma[i, t].X > 0.5 and (t == 1 or gamma[i, t - 1].X < 0.5):
                completed_orders.append(i)
        if completed_orders:
            log_lines.append(f"  [流转指令] 订单 {completed_orders} 拣选完毕，流转出库并释放打包台")
            slot_has_action = True

        # 仅打印有实际动作的 Slot
        if slot_has_action:
            print(f"\n▶▶ Slot {t} (时间: {(t - 1) * sigma}s - {t * sigma}s)")
            for line in log_lines:
                print(line)

else:
    print(f"求解失败，无可行解 (Status Code: {model.Status})。")