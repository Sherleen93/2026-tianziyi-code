import pandas as pd
import random
import copy
import time
import math
import os
from typing import List, Dict, Set, Any, Optional, Tuple
from itertools import product
import matplotlib.pyplot as plt

# ================= 1. 全局参数设定 =================
PI = 120
SIGMA = 20
A = 10
B = 10
C = 10
BW = 5
K_FILTER = 12
RANDOM_SEED = 3

# ================= 2. 实验总控参数 =================
# 原代码固定的是迭代次数 MAX_ITERATIONS = 250；
# 现在改为固定运行时间，其他公共参数、数据集、seed 和算法逻辑保持不变。
TIME_LIMIT_SECONDS = 300
SEEDS = [3, 22, 35, 42, 66]

OUTPUT_DIR = 'comparison_outputs_time'
PLOT_FILENAME = 'three_algorithms_mean_best_cost_vs_time.png'
SUMMARY_CSV = 'three_algorithms_summary_mean_final_cost_runtime_time_limited.csv'
DETAIL_CSV = 'three_algorithms_runs_detailed_time_limited.csv'
DETAILED_ITERATION_CSV = 'three_algorithms_iteration_records_time_limited.csv'
MEAN_CURVE_CSV = 'three_algorithms_mean_best_cost_curve_time_limited.csv'

# 时间曲线对齐点数。101 表示从 0 到 TIME_LIMIT_SECONDS 等距取 101 个点。
TIME_GRID_POINTS = 101

# ================= 批量数据集设置 =================
# 数据集文件夹需要和本代码放在同一目录下，例如：
# 50_time/orders.csv, 50_time/order_details.csv, 50_time/products.csv, 50_time/inventory.csv
DATASET_SIZES = [50]
DATASET_FOLDER_TEMPLATE = '{size}_time'
OUTPUT_DIR_TEMPLATE = '{size}_comparison_outputs_time'


class TimeLimitReached(Exception):
    """Raised internally when the time budget is reached during one DP evaluation."""
    pass


def check_time_limit(deadline: Optional[float] = None):
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeLimitReached


def safe_len(x) -> int:
    try:
        return len(x)
    except TypeError:
        return 0


def zone_remaining_count(rem_dict: Dict[int, Dict[int, Any]], oid: int, zone: int) -> int:
    return safe_len(rem_dict.get(oid, {}).get(zone, []))


def order_remaining_count(rem_dict: Dict[int, Dict[int, Any]], oid: int) -> int:
    return sum(zone_remaining_count(rem_dict, oid, z) for z in ZONES)

# y轴可手动设置；不想手动控制就保持 None。
# 例如：PLOT_Y_MIN = 105000; PLOT_Y_MAX = 180000; PLOT_Y_TICK_START = 115000; PLOT_Y_STEP = 10000
PLOT_Y_MIN = None
PLOT_Y_MAX = None
PLOT_Y_TICK_START = None
PLOT_Y_STEP = None

# ================= VNS 参数（主算法） =================
# 算子保持不变；VNS 外层采用“shaking + local search + neighborhood change”的标准流程。
VNS_RIDE_SHARE_WINDOW = 50
VNS_RIDE_SHARE_REQUIRE = 'either'      # {'sku','rack','either','both'}

# ================= SA 参数（参考文献式 SA-DPH） =================
# 参考 27-slot 文章中的 SA 设置：
# 1) 初始温度由初始解自适应确定；
# 2) 一个比初始解差 w% 的解，在初始温度下有 50% 的接受概率；
# 3) 每个温度下执行一个 epoch，初始 epoch 长度 K=10；
# 4) 每个 epoch 结束后根据 f_min 与 f_max 更新 K；
# 5) 每个 epoch 结束后降温，alpha=0.9；
# 6) 终止准则为温度低于 0.01 或运行时间耗尽。
SA_WORSE_RATIO = 0.05
SA_INITIAL_ACCEPT_PROB = 0.5
SA_COOLING_RATE = 0.9
SA_MIN_TEMPERATURE = 0.01
SA_INITIAL_EPOCH_LENGTH = 10

# 参考文献中的 SA 三个位置算子等概率选择：swap / shift / inversion。
SA_OPERATOR_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)

# ================= TS 参数（与 SA 使用相同位置算子） =================
# TS-DPH 的邻域算子调整为与参考文献式 SA-DPH 一致：swap / shift / inversion。
# 二者的区别仅保留在搜索框架上：SA 使用温度接受机制，TS 使用 tabu list、候选池和 aspiration。
TS_TABU_TENURE = 10
TS_ASPIRATION = True
TS_OPERATOR_WEIGHTS = (0.4, 0.4, 0.2)
TS_CANDIDATE_POOL = 3


# ================= 3. 数据加载与哈希预计算 =================
def load_data(data_dir: str = '.'):
    """Load one instance dataset from data_dir.

    Expected files in each folder:
    - orders.csv
    - order_details.csv
    - products.csv
    - inventory.csv
    """
    try:
        orders_df = pd.read_csv(os.path.join(data_dir, 'orders.csv'))
        details_df = pd.read_csv(os.path.join(data_dir, 'order_details.csv'))
        products_df = pd.read_csv(os.path.join(data_dir, 'products.csv'))
        inventory_df = pd.read_csv(os.path.join(data_dir, 'inventory.csv'))
    except FileNotFoundError as exc:
        print(f'[Data warning] Missing CSV file in {data_dir}: {exc}')
        return None, None, None, None, None

    order_info = orders_df.set_index('order_id').to_dict('index')
    sku_to_zone = products_df.set_index('sku_id')['zone_id'].to_dict()
    zones = sorted(list(set(sku_to_zone.values())))

    order_needs = {}
    for _, row in details_df.iterrows():
        oid, sku = int(row['order_id']), int(row['sku_id'])
        z = sku_to_zone.get(sku)
        if z is None:
            continue
        if oid not in order_needs:
            order_needs[oid] = {zn: set() for zn in zones}
        order_needs[oid][z].add(sku)

    rack_inv = {z: {} for z in zones}
    for _, row in inventory_df.iterrows():
        rid, sku, z = int(row['rack_id']), int(row['sku_id']), int(row['zone_id'])
        if rid not in rack_inv[z]:
            rack_inv[z][rid] = set()
        rack_inv[z][rid].add(sku)

    order_demands = {oid: sum(len(skus) for skus in zs.values()) for oid, zs in order_needs.items()}
    return order_info, order_needs, rack_inv, zones, order_demands


# These globals are refreshed before running each instance size.
ORDER_INFO = {}
ORDER_NEEDS_MASTER = {}
RACK_INV = {}
ZONES = []
ORDER_TOTAL_DEMANDS = {}
ORDER_SKU_CACHE = {}
ORDER_RACKS_CACHE = {}
CURRENT_DATA_DIR = None
CURRENT_INSTANCE_SIZE = None


def initialize_data(data_dir: str, instance_size: Optional[int] = None) -> bool:
    """Load one dataset and refresh all global caches used by DPH and metaheuristics."""
    global ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS
    global ORDER_SKU_CACHE, ORDER_RACKS_CACHE, CURRENT_DATA_DIR, CURRENT_INSTANCE_SIZE

    data = load_data(data_dir)
    if data[0] is None:
        return False

    ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS = data
    ORDER_SKU_CACHE = {}
    ORDER_RACKS_CACHE = {}

    for oid, zones_dict in ORDER_NEEDS_MASTER.items():
        ORDER_SKU_CACHE[oid] = set().union(*zones_dict.values()) if zones_dict else set()

        racks = set()
        for z, skus in zones_dict.items():
            if not skus:
                continue
            for sku in skus:
                for rid, rack_skus in RACK_INV[z].items():
                    if sku in rack_skus:
                        racks.add((z, rid))
        ORDER_RACKS_CACHE[oid] = racks

    CURRENT_DATA_DIR = data_dir
    CURRENT_INSTANCE_SIZE = instance_size
    return True


# ================= 4. 底层 DP（完全保留原逻辑） =================
def evaluate_dp(sequence: List[int], return_details: bool = False, deadline: Optional[float] = None):
    init_active = sequence[:C]
    init_rem = {oid: copy.deepcopy(ORDER_NEEDS_MASTER[oid]) for oid in init_active}

    initial_state = {
        'id': 'S_0_0', 'rem': init_rem, 'active': init_active, 'next_pos': len(init_active),
        'last_racks': {z: -1 for z in ZONES}, 'slot_count': 0, 'completed_slots': {},
        'trajectory': [], 'total_picked_items': 0, 'total_switches': 0
    }

    frontier = [initial_state]
    slot_index = 1
    best_final = None

    while frontier:
        check_time_limit(deadline)
        if all(not st['active'] and st['next_pos'] >= len(sequence) for st in frontier):
            def get_objective(x):
                return sum(ORDER_INFO[o]['weight'] * ((x['completed_slots'][o] - 1) * SIGMA + PI) for o in sequence)

            best_final = min(frontier, key=get_objective)
            break

        next_candidates = []
        for st in frontier:
            check_time_limit(deadline)
            if not st['active'] and st['next_pos'] >= len(sequence):
                next_candidates.append(((0, 0), st, None))
                continue

            zone_candidates = {z: [] for z in ZONES}
            for z in ZONES:
                active_needs_z = set().union(*[st['rem'][o][z] for o in st['active']])
                if not active_needs_z:
                    zone_candidates[z].append((None, 0, 0))
                else:
                    for rid, skus in RACK_INV[z].items():
                        is_sw = 1 if rid != st['last_racks'][z] else 0
                        max_q = (SIGMA - B * is_sw) // A
                        if max_q <= 0:
                            continue
                        overlap_count = len(skus & active_needs_z)
                        if overlap_count > 0:
                            zone_candidates[z].append((rid, min(overlap_count, max_q), is_sw))

                    zone_candidates[z].sort(key=lambda x: x[1], reverse=True)
                    zone_candidates[z] = zone_candidates[z][:K_FILTER]
                    if not zone_candidates[z]:
                        zone_candidates[z].append((None, 0, 0))

            actions = list(product(*[zone_candidates[z] for z in ZONES]))

            for action in actions:
                check_time_limit(deadline)
                rem_temp = {o: {zn: list(st['rem'][o][zn]) for zn in ZONES} for o in st['active']}
                step_picked = 0

                for i, z in enumerate(ZONES):
                    rid = action[i][0]
                    if rid is None:
                        continue
                    is_sw = action[i][2]
                    q_limit = (SIGMA - B * is_sw) // A

                    sorted_active = sorted(st['active'], key=lambda x: len(rem_temp[x][z]) if rem_temp[x][z] else 999)
                    for o in sorted_active:
                        if q_limit <= 0:
                            break
                        hits = set(rem_temp[o][z]) & RACK_INV[z][rid]
                        for sku in list(hits):
                            if q_limit <= 0:
                                break
                            rem_temp[o][z].remove(sku)
                            q_limit -= 1
                            step_picked += 1

                newly_done = [o for o in st['active'] if all(not rem_temp[o][zn] for zn in ZONES)]

                if step_picked == 0 and not newly_done:
                    continue

                total_done_orders = len(st['completed_slots']) + len(newly_done)
                total_picked_items = st['total_picked_items'] + step_picked
                next_candidates.append(((total_done_orders, total_picked_items), st, action))

        if not next_candidates:
            break

        for idx in range(len(next_candidates)):
            score, st, action = next_candidates[idx]
            rem_temp_eval = {o: {zn: list(st['rem'][o][zn]) for zn in ZONES} for o in st['active']}
            step_switches = sum(action[i][2] for i in range(len(ZONES))) if action else 0

            if action:
                for i, z in enumerate(ZONES):
                    if action[i][0] is None:
                        continue
                    q_limit = (SIGMA - B * action[i][2]) // A
                    for o in sorted(st['active'],
                                    key=lambda x: zone_remaining_count(rem_temp_eval, x, z) if zone_remaining_count(rem_temp_eval, x, z) > 0 else 999):
                        if q_limit <= 0:
                            break
                        hits = set(rem_temp_eval[o][z]) & RACK_INV[z][action[i][0]]
                        for sku in list(hits):
                            if q_limit <= 0:
                                break
                            rem_temp_eval[o][z].remove(sku)
                            q_limit -= 1

            active_after_action = [o for o in st['active'] if order_remaining_count(rem_temp_eval, o) > 0]
            rem_counts_after_action = [order_remaining_count(rem_temp_eval, o) for o in active_after_action]
            min_rem_single = min(rem_counts_after_action) if rem_counts_after_action else 0
            next_candidates[idx] = ((*score, -min_rem_single, -(st.get('total_switches', 0) + step_switches)), st,
                                    action)

        next_candidates.sort(key=lambda x: x[0], reverse=True)

        next_frontier = []
        visited_states = set()
        kept_count = 0

        for fine_score, st, action in next_candidates:
            if kept_count >= BW:
                break
            if action is None:
                next_frontier.append(st)
                kept_count += 1
                continue

            rem_temp = {o: {zn: list(st['rem'][o][zn]) for zn in ZONES} for o in st['active']}
            new_racks = {z: st['last_racks'][z] for z in ZONES}

            for i, z in enumerate(ZONES):
                if action[i][0] is None:
                    continue
                new_racks[z] = action[i][0]
                q_limit = (SIGMA - B * action[i][2]) // A
                for o in sorted(st['active'], key=lambda x: len(rem_temp[x][z]) if rem_temp[x][z] else 999):
                    if q_limit <= 0:
                        break
                    hits = set(rem_temp[o][z]) & RACK_INV[z][action[i][0]]
                    for sku in list(hits):
                        if q_limit <= 0:
                            break
                        rem_temp[o][z].remove(sku)
                        q_limit -= 1

            state_signature = (
                tuple(sorted(st['active'])),
                tuple((o, tuple((z, tuple(sorted(rem_temp[o][z]))) for z in ZONES)) for o in sorted(st['active'])),
                tuple((z, new_racks[z]) for z in ZONES)
            )

            if state_signature in visited_states:
                continue
            visited_states.add(state_signature)

            new_st = copy.deepcopy(st)
            new_st['slot_count'] += 1
            new_st['id'] = f"S_{slot_index}_{kept_count + 1}"
            new_st['total_switches'] = new_st.get('total_switches', 0) + sum(action[i][2] for i in range(len(ZONES)))

            if return_details:
                step_log = {'slot': new_st['slot_count'], 'active_at_start': list(new_st['active']), 'racks': {},
                            'detailed_skus': {z: [] for z in ZONES}}
                for i, z in enumerate(ZONES):
                    rid = action[i][0]
                    step_log['racks'][z] = rid
                    if rid is not None:
                        new_st['last_racks'][z] = rid
                        q_limit = (SIGMA - B * action[i][2]) // A
                        for o in sorted(new_st['active'],
                                        key=lambda x: len(new_st['rem'][x][z]) if new_st['rem'][x][z] else 999):
                            if q_limit <= 0:
                                break
                            hits = new_st['rem'][o][z] & RACK_INV[z][rid]
                            if hits:
                                for sku in list(hits):
                                    if q_limit <= 0:
                                        break
                                    new_st['rem'][o][z].remove(sku)
                                    q_limit -= 1
                                    new_st['total_picked_items'] += 1
                                    step_log['detailed_skus'][z].append(f"订单{o}(SKU:{sku})")
                new_st['trajectory'].append(step_log)
            else:
                for i, z in enumerate(ZONES):
                    rid = action[i][0]
                    if rid is not None:
                        new_st['last_racks'][z] = rid
                        q_limit = (SIGMA - B * action[i][2]) // A
                        for o in sorted(new_st['active'],
                                        key=lambda x: len(new_st['rem'][x][z]) if new_st['rem'][x][z] else 999):
                            if q_limit <= 0:
                                break
                            hits = new_st['rem'][o][z] & RACK_INV[z][rid]
                            if hits:
                                for sku in list(hits):
                                    if q_limit <= 0:
                                        break
                                    new_st['rem'][o][z].remove(sku)
                                    q_limit -= 1
                                    new_st['total_picked_items'] += 1

            done_this_slot = [o for o in new_st['active'] if all(not new_st['rem'][o][zn] for zn in ZONES)]
            for o in done_this_slot:
                new_st['active'].remove(o)
                new_st['completed_slots'][o] = new_st['slot_count']

            while len(new_st['active']) < C and new_st['next_pos'] < len(sequence):
                new_oid = sequence[new_st['next_pos']]
                new_st['active'].append(new_oid)
                new_st['rem'][new_oid] = copy.deepcopy(ORDER_NEEDS_MASTER[new_oid])
                new_st['next_pos'] += 1

            next_frontier.append(new_st)
            kept_count += 1

        frontier = next_frontier
        slot_index += 1

    if best_final is None:
        if return_details:
            return float('inf'), None
        return float('inf')

    total_cost = sum(ORDER_INFO[o]['weight'] * ((best_final['completed_slots'][o] - 1) * SIGMA + PI) for o in sequence)
    if return_details:
        return total_cost, best_final
    return total_cost


# ================= 5. 共用工具与算子 =================
def build_initial_sequence() -> List[int]:
    return sorted(ORDER_INFO.keys(), key=lambda oid: (-ORDER_INFO[oid]['weight'], -ORDER_TOTAL_DEMANDS[oid]))


def make_cost_evaluator(deadline: Optional[float] = None):
    dp_cache: Dict[Tuple[int, ...], float] = {}

    def get_cost(seq: List[int]) -> float:
        seq_tuple = tuple(seq)
        if seq_tuple not in dp_cache:
            check_time_limit(deadline)
            dp_cache[seq_tuple] = evaluate_dp(list(seq), return_details=False, deadline=deadline)
        return dp_cache[seq_tuple]

    return get_cost, dp_cache


def choose_operator(rng: random.Random, weights: Tuple[float, float, float]) -> int:
    r = rng.random()
    cum = 0.0
    for idx, w in enumerate(weights, start=1):
        cum += w
        if r <= cum:
            return idx
    return 3


def has_shared_structure(candidate_oid: int, neighbor_oids: List[int], require_mode: str = 'either') -> bool:
    if not neighbor_oids:
        return False
    neighbor_skus = set()
    neighbor_racks = set()
    for oid in neighbor_oids:
        neighbor_skus |= ORDER_SKU_CACHE[oid]
        neighbor_racks |= ORDER_RACKS_CACHE[oid]
    sku_shared = len(ORDER_SKU_CACHE[candidate_oid] & neighbor_skus) > 0
    rack_shared = len(ORDER_RACKS_CACHE[candidate_oid] & neighbor_racks) > 0
    if require_mode == 'sku':
        return sku_shared
    if require_mode == 'rack':
        return rack_shared
    if require_mode == 'both':
        return sku_shared and rack_shared
    return sku_shared or rack_shared


def op_n1_ride_sharing(seq: List[int], rng: random.Random,
                       window_size: int = 1,
                       require_mode: str = 'either',
                       aggressive: bool = False,
                       label: str = 'N1_ride_sharing') -> Tuple[List[int], Dict[str, Any]]:
    s = seq[:]
    n = len(s)
    if n < 2:
        return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

    i = rng.randrange(1, n)
    candidate_oid = s[i]
    matched_positions = []
    window_size = max(1, int(window_size))

    for j in range(i - 1, -1, -1):
        left = max(0, j - window_size + 1)
        window_oids = s[left:j + 1]
        if has_shared_structure(candidate_oid, window_oids, require_mode=require_mode):
            matched_positions.append(j)

    if not matched_positions:
        return s, {'operator': label, 'move_applied': False, 'reason': 'no_match', 'from_index': i}

    matched_j = rng.choice(matched_positions) if aggressive else matched_positions[0]
    item = s.pop(i)
    s.insert(matched_j + 1, item)
    return s, {
        'operator': label,
        'move_applied': s != seq,
        'from_index': i,
        'to_after_index': matched_j,
        'matched_count': len(matched_positions),
    }


def op_n2_shift(seq: List[int], rng: random.Random, max_block_len: int, label: str = 'N2_shift') -> Tuple[List[int], Dict[str, Any]]:
    s = seq[:]
    n = len(s)
    if n < 3:
        return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

    max_block_len = max(1, min(int(max_block_len), n - 1))
    block_len = rng.randint(1, max_block_len)
    a = rng.randint(0, n - block_len)
    b = a + block_len - 1

    valid_c = list(range(-1, a)) + list(range(b + 1, n - 1))
    if not valid_c:
        return s, {'operator': label, 'move_applied': False, 'reason': 'no_valid_target'}
    c = rng.choice(valid_c)

    block = s[a:b + 1]
    remaining = s[:a] + s[b + 1:]
    insert_pos = c + 1 if c < a else c - block_len + 1
    insert_pos = max(0, min(insert_pos, len(remaining)))
    moved = remaining[:insert_pos] + block + remaining[insert_pos:]
    return moved, {
        'operator': label,
        'move_applied': moved != seq,
        'block_len': block_len,
        'block_start': a,
        'block_end': b,
        'target_after_index': c,
    }


def op_n3_reverse(seq: List[int], rng: random.Random, max_rev_len: int, label: str = 'N3_reverse') -> Tuple[List[int], Dict[str, Any]]:
    s = seq[:]
    n = len(s)
    if n < 2:
        return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

    max_rev_len = max(2, min(int(max_rev_len), n))
    rev_len = rng.randint(2, max_rev_len)
    a = rng.randint(0, n - rev_len)
    b = a + rev_len - 1
    s[a:b + 1] = list(reversed(s[a:b + 1]))
    return s, {
        'operator': label,
        'move_applied': s != seq,
        'reverse_len': rev_len,
        'reverse_start': a,
        'reverse_end': b,
    }


def propose_neighbor(seq: List[int], rng: random.Random, operator_id: int,
                     shift_cap: int, reverse_cap: int,
                     ride_window: int = 1, ride_mode: str = 'either', aggressive_n1: bool = False,
                     label_prefix: str = '') -> Tuple[List[int], Dict[str, Any]]:
    if operator_id == 1:
        return op_n1_ride_sharing(seq, rng, ride_window, ride_mode, aggressive_n1,
                                  label=f'{label_prefix}N1_ride_sharing')
    if operator_id == 2:
        return op_n2_shift(seq, rng, shift_cap, label=f'{label_prefix}N2_shift')
    return op_n3_reverse(seq, rng, reverse_cap, label=f'{label_prefix}N3_reverse')

def build_tabu_signature(meta: Dict[str, Any]) -> Tuple:
    """Build a tabu signature for the move used to generate a candidate sequence.

    The current TS-DPH uses the same position-based operators as SA-DPH:
    swap, shift, and inversion. The older N1/N2/N3 signatures are retained
    for compatibility with other parts of the code, but TS now mainly uses
    the swap/shift/inversion branches.
    """
    op = meta.get('operator', '')
    op_lower = op.lower()

    # TS / SA reference-style operators
    if 'swap' in op_lower:
        i = meta.get('swap_i', -1)
        j = meta.get('swap_j', -1)
        return ('swap', min(i, j), max(i, j))

    if 'shift' in op_lower and 'shift_start' in meta:
        return (
            'shift',
            meta.get('shift_start', -1),
            meta.get('shift_end', -1),
            meta.get('shift_after', -1)
        )

    if 'inversion' in op_lower:
        return (
            'inversion',
            meta.get('inversion_start', -1),
            meta.get('inversion_end', -1)
        )

    # Compatibility with the original problem-specific N1/N2/N3 operators
    if 'N1' in op:
        return (
            'N1',
            meta.get('from_index', -1),
            meta.get('to_after_index', -1)
        )

    if 'N2' in op:
        return (
            'N2',
            meta.get('block_start', -1),
            meta.get('block_end', -1),
            meta.get('target_after_index', -1)
        )

    if 'N3' in op:
        return (
            'N3',
            meta.get('reverse_start', -1),
            meta.get('reverse_end', -1)
        )

    return (op,)

def acceptance_probability(delta: float, temperature: float) -> float:
    if temperature <= 0:
        return 0.0
    try:
        return math.exp(-delta / temperature)
    except OverflowError:
        return 0.0



# ================= 6. 三种算法（运行时间停止准则版本） =================
def vns_optimize(time_limit_seconds: float = TIME_LIMIT_SECONDS,
                 quiet: bool = True,
                 random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    VNS-DPH（运行时间停止准则）：
    1) 外层 VNS 搜索订单投产序列；
    2) 内层 evaluate_dp() 作为 DPH 评价器；
    3) 算子保持不变：N1 ride-sharing, N2 shift, N3 reverse；
    4) 不再设置单独的 SHAKE 状态，也不区分局部/大范围参数；
    5) 接受准则：仅接受改进解。
    """
    start_time = time.perf_counter()
    deadline = start_time + time_limit_seconds
    rng = random.Random(random_seed)
    n_orders = len(ORDER_INFO)
    get_cost, dp_cache = make_cost_evaluator(deadline=deadline)
    K_MAX = 3

    def elapsed() -> float:
        return time.perf_counter() - start_time

    def time_remaining() -> bool:
        return elapsed() < time_limit_seconds

    # 不再区分小范围/大范围；仅保留算子合法范围。
    shift_cap = max(1, n_orders - 1)
    reverse_cap = max(2, n_orders)

    def apply_neighborhood(seq: List[int], k: int, phase: str) -> Tuple[List[int], Dict[str, Any]]:
        if k == 1:
            cand_seq, meta = op_n1_ride_sharing(
                seq, rng, VNS_RIDE_SHARE_WINDOW, VNS_RIDE_SHARE_REQUIRE,
                aggressive=False, label=f'VNS_{phase}_N1'
            )
        elif k == 2:
            cand_seq, meta = op_n2_shift(seq, rng, shift_cap, label=f'VNS_{phase}_N2')
        elif k == 3:
            cand_seq, meta = op_n3_reverse(seq, rng, reverse_cap, label=f'VNS_{phase}_N3')
        else:
            raise ValueError(f'Invalid neighborhood index: {k}')

        meta['phase'] = phase
        meta['neighborhood_k'] = k
        return cand_seq, meta

    def local_search(seq: List[int], cost: float) -> Tuple[List[int], float, int, List[str]]:
        """VND-style local search using N1, N2, and N3."""
        local_seq = seq[:]
        local_cost = cost
        st = 1
        local_steps = 0
        local_ops = []

        while st <= K_MAX and time_remaining():
            cand_seq, meta = apply_neighborhood(local_seq, st, phase='local')
            try:
                cand_cost = get_cost(cand_seq)
            except TimeLimitReached:
                break

            local_steps += 1
            local_ops.append(meta.get('operator', ''))

            if cand_cost < local_cost:
                local_seq = cand_seq[:]
                local_cost = cand_cost
                st = 1
            else:
                st += 1

        return local_seq, local_cost, local_steps, local_ops

    best_seq = build_initial_sequence()
    try:
        initial_cost = get_cost(best_seq)
    except TimeLimitReached:
        initial_cost = float('inf')

    best_cost = initial_cost
    current_seq = best_seq[:]
    current_cost = best_cost

    cost_history = [best_cost]
    iteration_records = [{
        'iteration': 0,
        'best_cost': best_cost,
        'current_cost': current_cost,
        'candidate_cost': current_cost,
        'shaken_cost': current_cost,
        'operator': 'initial',
        'accepted_move': 0,
        'global_improved': 0,
        'neighborhood_k': 0,
        'local_search_steps': 0,
        'elapsed_seconds': 0.0,
        'dp_calls': 1,
        'dp_cache_size': len(dp_cache),
    }]

    iteration = 0

    while time_remaining():
        k = 1

        while k <= K_MAX and time_remaining():
            iteration += 1
            state_k = k

            # Step 1: shaking by the kth neighborhood
            shaken_seq, shake_meta = apply_neighborhood(current_seq, state_k, phase='shake')
            try:
                shaken_cost = get_cost(shaken_seq)
            except TimeLimitReached:
                break

            # Step 2: VND-style local search from the shaken sequence
            local_seq, local_cost, local_steps, local_ops = local_search(shaken_seq, shaken_cost)

            accepted = 0
            improved = 0

            # Step 3: neighborhood change
            if local_cost < current_cost:
                current_seq = local_seq[:]
                current_cost = local_cost
                accepted = 1
                k = 1

                if current_cost < best_cost:
                    best_seq = current_seq[:]
                    best_cost = current_cost
                    improved = 1
            else:
                k += 1

            if not quiet:
                print(
                    f"[VNS][Seed {random_seed}][Iter {iteration:04d}] "
                    f"k={state_k} shake={shake_meta.get('operator')} "
                    f"shaken={shaken_cost} local={local_cost} "
                    f"cur={current_cost} best={best_cost} local_steps={local_steps}"
                )

            cost_history.append(best_cost)
            iteration_records.append({
                'iteration': iteration,
                'best_cost': best_cost,
                'current_cost': current_cost,
                'candidate_cost': local_cost,
                'shaken_cost': shaken_cost,
                'operator': shake_meta.get('operator', ''),
                'accepted_move': accepted,
                'global_improved': improved,
                'neighborhood_k': state_k,
                'local_search_steps': local_steps,
                'local_operators': ' + '.join(local_ops),
                'elapsed_seconds': elapsed(),
                'dp_calls': 1 + local_steps,
                'dp_cache_size': len(dp_cache),
            })

    total_elapsed = elapsed()
    return {
        'algorithm': 'VNS',
        'seed': random_seed,
        'initial_cost': initial_cost,
        'best_cost': best_cost,
        'best_seq': best_seq,
        'cost_history': cost_history,
        'iteration_records': iteration_records,
        'elapsed_seconds': total_elapsed,
        'actual_iterations': iteration,
        'time_limit_seconds': time_limit_seconds,
        'dp_cache_size': len(dp_cache),
    }



def op_sa_swap(seq: List[int], rng: random.Random,
               label: str = 'SA_swap') -> Tuple[List[int], Dict[str, Any]]:
    """Reference-style SA operator 1: swap two random positions."""
    s = seq[:]
    n = len(s)

    if n < 2:
        return s, {
            'operator': label,
            'move_applied': False,
            'reason': 'too_short'
        }

    i, j = rng.sample(range(n), 2)
    s[i], s[j] = s[j], s[i]

    return s, {
        'operator': label,
        'move_applied': s != seq,
        'swap_i': i,
        'swap_j': j
    }


def op_sa_shift(seq: List[int], rng: random.Random,
                label: str = 'SA_shift') -> Tuple[List[int], Dict[str, Any]]:
    """
    Reference-style SA operator 2: shift.
    Randomly select three positions a < b < c and move the segment [a, b]
    to the position after c.
    """
    s = seq[:]
    n = len(s)

    if n < 3:
        return s, {
            'operator': label,
            'move_applied': False,
            'reason': 'too_short'
        }

    a, b, c = sorted(rng.sample(range(n), 3))

    block = s[a:b + 1]
    remaining = s[:a] + s[b + 1:]

    # In the original sequence, the block is moved to after position c.
    # After removing [a, b], the corrected insertion position is:
    insert_pos = c - len(block) + 1
    insert_pos = max(0, min(insert_pos, len(remaining)))

    moved = remaining[:insert_pos] + block + remaining[insert_pos:]

    return moved, {
        'operator': label,
        'move_applied': moved != seq,
        'shift_start': a,
        'shift_end': b,
        'shift_after': c,
        'block_len': len(block)
    }


def op_sa_inversion(seq: List[int], rng: random.Random,
                    label: str = 'SA_inversion') -> Tuple[List[int], Dict[str, Any]]:
    """Reference-style SA operator 3: invert a random subsequence."""
    s = seq[:]
    n = len(s)

    if n < 2:
        return s, {
            'operator': label,
            'move_applied': False,
            'reason': 'too_short'
        }

    a, b = sorted(rng.sample(range(n), 2))
    s[a:b + 1] = list(reversed(s[a:b + 1]))

    return s, {
        'operator': label,
        'move_applied': s != seq,
        'inversion_start': a,
        'inversion_end': b
    }


def propose_sa_reference_neighbor(seq: List[int], rng: random.Random) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate a candidate order sequence for reference-style SA-DPH.
    The three position-based operators are selected with equal probability.
    """
    op_id = choose_operator(rng, SA_OPERATOR_WEIGHTS)

    if op_id == 1:
        return op_sa_swap(seq, rng, label='SA_swap')
    if op_id == 2:
        return op_sa_shift(seq, rng, label='SA_shift')
    return op_sa_inversion(seq, rng, label='SA_inversion')



def propose_ts_reference_neighbor(seq: List[int], rng: random.Random) -> Tuple[List[int], Dict[str, Any]]:
    """
    Generate a candidate order sequence for TS-DPH using the same three
    position-based operators as SA-DPH: swap, shift, and inversion.
    """
    op_id = choose_operator(rng, TS_OPERATOR_WEIGHTS)

    if op_id == 1:
        return op_sa_swap(seq, rng, label='TS_swap')
    if op_id == 2:
        return op_sa_shift(seq, rng, label='TS_shift')
    return op_sa_inversion(seq, rng, label='TS_inversion')

def sa_optimize(time_limit_seconds: float = TIME_LIMIT_SECONDS,
                quiet: bool = True,
                random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    Reference-style SA-DPH（运行时间停止准则）：
    1) SA 流程和参数参考 27-slot 文章；
    2) 搜索对象仍为本文 IORP 的订单投产序列；
    3) 每个候选序列仍由 evaluate_dp() / DPH 评价；
    4) 目标函数仍为 total weighted completion time；
    5) 终止准则：温度低于 SA_MIN_TEMPERATURE 或运行时间耗尽。
    """
    start_time = time.perf_counter()
    deadline = start_time + time_limit_seconds
    rng = random.Random(random_seed)
    get_cost, dp_cache = make_cost_evaluator(deadline=deadline)

    def elapsed() -> float:
        return time.perf_counter() - start_time

    def time_remaining() -> bool:
        return elapsed() < time_limit_seconds

    best_seq = build_initial_sequence()

    try:
        initial_cost = get_cost(best_seq)
    except TimeLimitReached:
        initial_cost = float('inf')

    best_cost = initial_cost
    current_seq = best_seq[:]
    current_cost = best_cost

    # Reference-style initial temperature:
    # A solution that is SA_WORSE_RATIO worse than the initial solution
    # has SA_INITIAL_ACCEPT_PROB probability of being accepted initially.
    # The 27-slot paper uses f(X)/m for the multi-workstation rack-visit objective.
    # Here the same idea is adapted to the total weighted completion time objective.
    if math.isfinite(initial_cost) and initial_cost > 0:
        temperature = -SA_WORSE_RATIO * initial_cost / math.log(SA_INITIAL_ACCEPT_PROB)
    else:
        temperature = 1.0
    temperature = max(temperature, 1e-9)

    epoch_length = int(SA_INITIAL_EPOCH_LENGTH)

    cost_history = [best_cost]
    iteration_records = [{
        'iteration': 0,
        'epoch': 0,
        'epoch_length': epoch_length,
        'best_cost': best_cost,
        'current_cost': current_cost,
        'candidate_cost': current_cost,
        'operator': 'initial',
        'accepted_move': 0,
        'global_improved': 0,
        'temperature': temperature,
        'elapsed_seconds': 0.0,
        'dp_calls': 1,
        'dp_cache_size': len(dp_cache),
    }]

    iteration = 0
    epoch = 0

    while time_remaining() and temperature >= SA_MIN_TEMPERATURE:
        epoch += 1
        epoch_costs = []

        k = 1
        while k <= epoch_length and time_remaining():
            iteration += 1

            cand_seq, meta = propose_sa_reference_neighbor(current_seq, rng)

            try:
                cand_cost = get_cost(cand_seq)
            except TimeLimitReached:
                break

            delta = cand_cost - current_cost
            accepted = 0
            global_improved = 0

            if delta < 0:
                current_seq = cand_seq[:]
                current_cost = cand_cost
                accepted = 1
            else:
                prob = acceptance_probability(delta, temperature)
                if rng.random() < prob:
                    current_seq = cand_seq[:]
                    current_cost = cand_cost
                    accepted = 1

            if current_cost < best_cost:
                best_seq = current_seq[:]
                best_cost = current_cost
                global_improved = 1

            epoch_costs.append(current_cost)
            cost_history.append(best_cost)

            iteration_records.append({
                'iteration': iteration,
                'epoch': epoch,
                'epoch_length': epoch_length,
                'best_cost': best_cost,
                'current_cost': current_cost,
                'candidate_cost': cand_cost,
                'operator': meta.get('operator', ''),
                'accepted_move': accepted,
                'global_improved': global_improved,
                'temperature': temperature,
                'elapsed_seconds': elapsed(),
                'dp_calls': 1,
                'dp_cache_size': len(dp_cache),
                'meta': str(meta),
            })

            if not quiet:
                print(
                    f"[SA-Ref][Seed {random_seed}][Epoch {epoch:04d}][Iter {iteration:05d}] "
                    f"{meta.get('operator')} cand={cand_cost} cur={current_cost} "
                    f"best={best_cost} T={temperature:.6f} K={epoch_length} accept={accepted}"
                )

            k += 1

        if not epoch_costs:
            break

        f_min = min(epoch_costs)
        f_max = max(epoch_costs)

        # Reference-style epoch length update:
        # K := K + floor(K * (1 - exp((f_min - f_max) / f_max))).
        if f_max > 0 and math.isfinite(f_max):
            growth = math.floor(epoch_length * (1 - math.exp((f_min - f_max) / f_max)))
            epoch_length = max(1, epoch_length + growth)

        # Cooling after each epoch.
        temperature = temperature * SA_COOLING_RATE

    total_elapsed = elapsed()

    return {
        'algorithm': 'SA',
        'seed': random_seed,
        'initial_cost': initial_cost,
        'best_cost': best_cost,
        'best_seq': best_seq,
        'cost_history': cost_history,
        'iteration_records': iteration_records,
        'elapsed_seconds': total_elapsed,
        'actual_iterations': iteration,
        'actual_epochs': epoch,
        'time_limit_seconds': time_limit_seconds,
        'dp_cache_size': len(dp_cache),
        'used_params': {
            'sa_scheme': 'reference-style SA with DPH evaluator',
            'SA_WORSE_RATIO': SA_WORSE_RATIO,
            'SA_INITIAL_ACCEPT_PROB': SA_INITIAL_ACCEPT_PROB,
            'SA_COOLING_RATE': SA_COOLING_RATE,
            'SA_MIN_TEMPERATURE': SA_MIN_TEMPERATURE,
            'SA_INITIAL_EPOCH_LENGTH': SA_INITIAL_EPOCH_LENGTH,
            'neighborhoods': 'swap, shift, inversion',
            'stopping_criterion': 'temperature threshold or CPU time limit',
        }
    }


def ts_optimize(time_limit_seconds: float = TIME_LIMIT_SECONDS,
                quiet: bool = True,
                random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    TS-DPH（运行时间停止准则，使用与 SA-DPH 相同的邻域算子）：
    1) 外层 TS 搜索订单投产序列；
    2) 内层 evaluate_dp() / DPH 评价每个候选序列；
    3) 邻域算子与参考文献式 SA-DPH 保持一致：swap、shift、inversion；
    4) TS 与 SA 的区别体现在搜索机制上：TS 使用 candidate pool、tabu list 和 aspiration；
    5) 停止准则为统一 CPU time limit。
    """
    start_time = time.perf_counter()
    deadline = start_time + time_limit_seconds
    rng = random.Random(random_seed)
    get_cost, dp_cache = make_cost_evaluator(deadline=deadline)

    def elapsed() -> float:
        return time.perf_counter() - start_time

    def time_remaining() -> bool:
        return elapsed() < time_limit_seconds

    best_seq = build_initial_sequence()

    try:
        initial_cost = get_cost(best_seq)
    except TimeLimitReached:
        initial_cost = float('inf')

    best_cost = initial_cost
    current_seq = best_seq[:]
    current_cost = best_cost

    tabu_list = []

    cost_history = [best_cost]
    iteration_records = [{
        'iteration': 0,
        'best_cost': best_cost,
        'current_cost': current_cost,
        'candidate_cost': current_cost,
        'operator': 'initial',
        'accepted_move': 0,
        'global_improved': 0,
        'is_tabu': 0,
        'aspiration_used': 0,
        'tabu_size': 0,
        'candidate_pool_size': 0,
        'elapsed_seconds': 0.0,
        'dp_calls': 1,
        'dp_cache_size': len(dp_cache),
    }]

    iteration = 0

    while time_remaining():
        iteration += 1
        candidate_pool = []

        for _ in range(TS_CANDIDATE_POOL):
            if not time_remaining():
                break

            cand_seq, meta = propose_ts_reference_neighbor(current_seq, rng)

            try:
                cand_cost = get_cost(cand_seq)
            except TimeLimitReached:
                break

            tabu_signature = build_tabu_signature(meta)
            is_tabu = tabu_signature in tabu_list

            # Aspiration criterion: 即使该 move 在 tabu list 中，
            # 只要能产生新的全局最优解，也允许被选择。
            aspiration_used = bool(TS_ASPIRATION and is_tabu and cand_cost < best_cost)
            admissible = (not is_tabu) or aspiration_used

            candidate_pool.append({
                'seq': cand_seq,
                'meta': meta,
                'cost': cand_cost,
                'tabu_signature': tabu_signature,
                'is_tabu': is_tabu,
                'aspiration_used': aspiration_used,
                'admissible': admissible,
            })

        if not candidate_pool:
            break

        admissible_candidates = [x for x in candidate_pool if x['admissible']]

        # 标准 TS 逻辑：优先从 admissible candidates 中选择最好者；
        # 若全为 tabu 且没有 aspiration，则退而选择候选池中目标值最好的一个，
        # 避免搜索停滞。
        if admissible_candidates:
            chosen = min(admissible_candidates, key=lambda x: x['cost'])
        else:
            chosen = min(candidate_pool, key=lambda x: x['cost'])

        current_seq = chosen['seq'][:]
        current_cost = chosen['cost']

        tabu_list.append(chosen['tabu_signature'])
        if len(tabu_list) > TS_TABU_TENURE:
            tabu_list.pop(0)

        global_improved = 0
        if current_cost < best_cost:
            best_seq = current_seq[:]
            best_cost = current_cost
            global_improved = 1

        if not quiet:
            print(
                f"[TS][Seed {random_seed}][Iter {iteration:04d}] "
                f"{chosen['meta'].get('operator')} "
                f"cand={chosen['cost']} cur={current_cost} best={best_cost} "
                f"tabu={chosen['is_tabu']} aspiration={chosen['aspiration_used']} "
                f"pool={len(candidate_pool)}"
            )

        cost_history.append(best_cost)
        iteration_records.append({
            'iteration': iteration,
            'best_cost': best_cost,
            'current_cost': current_cost,
            'candidate_cost': chosen['cost'],
            'operator': chosen['meta'].get('operator', ''),
            'accepted_move': 1,
            'global_improved': global_improved,
            'is_tabu': int(chosen['is_tabu']),
            'aspiration_used': int(chosen['aspiration_used']),
            'tabu_size': len(tabu_list),
            'candidate_pool_size': len(candidate_pool),
            'elapsed_seconds': elapsed(),
            'dp_calls': len(candidate_pool),
            'dp_cache_size': len(dp_cache),
            'meta': str(chosen['meta']),
        })

    total_elapsed = elapsed()
    return {
        'algorithm': 'TS',
        'seed': random_seed,
        'initial_cost': initial_cost,
        'best_cost': best_cost,
        'best_seq': best_seq,
        'cost_history': cost_history,
        'iteration_records': iteration_records,
        'elapsed_seconds': total_elapsed,
        'actual_iterations': iteration,
        'time_limit_seconds': time_limit_seconds,
        'dp_cache_size': len(dp_cache),
        'used_params': {
            'ts_scheme': 'tabu search with SA-style swap/shift/inversion operators and DPH evaluator',
            'TS_TABU_TENURE': TS_TABU_TENURE,
            'TS_ASPIRATION': TS_ASPIRATION,
            'TS_OPERATOR_WEIGHTS': TS_OPERATOR_WEIGHTS,
            'TS_CANDIDATE_POOL': TS_CANDIDATE_POOL,
            'neighborhoods': 'swap, shift, inversion',
            'stopping_criterion': 'CPU time limit',
        }
    }


# ================= 7. 多次运行、均值、绘图、CSV（时间横轴） =================
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_algorithm_five_seeds(algorithm_name: str, optimize_func, seeds: List[int],
                             time_limit_seconds: float, quiet: bool = True):
    results = []
    for seed in seeds:
        res = optimize_func(time_limit_seconds=time_limit_seconds, quiet=quiet, random_seed=seed)
        results.append(res)
    return results


def build_time_based_mean_curve(all_results: Dict[str, List[Dict[str, Any]]],
                                time_limit_seconds: float,
                                grid_points: int = TIME_GRID_POINTS) -> pd.DataFrame:
    """
    固定运行时间停止准则下，不同算法/不同seed的迭代次数可能不同。
    因此这里先把每条 best-so-far 曲线对齐到统一时间网格，再求5次运行均值。
    """
    if grid_points < 2:
        grid_points = 2

    time_grid = [time_limit_seconds * i / (grid_points - 1) for i in range(grid_points)]
    aligned_rows = []

    for algo_name, run_results in all_results.items():
        for r in run_results:
            records = sorted(r['iteration_records'], key=lambda x: x['elapsed_seconds'])
            times = [float(x['elapsed_seconds']) for x in records]
            costs = [float(x['best_cost']) for x in records]

            idx = 0
            last_cost = costs[0]

            for t in time_grid:
                while idx < len(times) and times[idx] <= t:
                    last_cost = costs[idx]
                    idx += 1

                aligned_rows.append({
                    'algorithm': algo_name,
                    'seed': r['seed'],
                    'elapsed_seconds': t,
                    'best_cost': last_cost,
                })

    aligned_df = pd.DataFrame(aligned_rows)

    mean_df = (
        aligned_df
        .groupby(['algorithm', 'elapsed_seconds'], as_index=False)
        .agg(
            mean_best_cost=('best_cost', 'mean'),
            std_best_cost=('best_cost', 'std'),
            min_best_cost=('best_cost', 'min'),
            max_best_cost=('best_cost', 'max'),
        )
    )

    return mean_df.sort_values(by=['algorithm', 'elapsed_seconds'])


def build_summary_tables(all_results: Dict[str, List[Dict[str, Any]]],
                         time_limit_seconds: float):
    summary_rows = []
    detail_rows = []

    for algo_name, run_results in all_results.items():
        mean_final_cost = sum(r['best_cost'] for r in run_results) / len(run_results)
        mean_runtime = sum(r['elapsed_seconds'] for r in run_results) / len(run_results)
        mean_iterations = sum(r['actual_iterations'] for r in run_results) / len(run_results)

        summary_rows.append({
            'algorithm': algo_name,
            'time_limit_seconds': time_limit_seconds,
            'mean_final_best_cost': mean_final_cost,
            'mean_total_runtime_seconds': mean_runtime,
            'mean_actual_iterations': mean_iterations,
        })

        for r in run_results:
            detail_rows.append({
                'algorithm': algo_name,
                'seed': r['seed'],
                'initial_cost': r['initial_cost'],
                'final_best_cost': r['best_cost'],
                'total_runtime_seconds': r['elapsed_seconds'],
                'actual_iterations': r['actual_iterations'],
                'time_limit_seconds': time_limit_seconds,
                'dp_cache_size': r['dp_cache_size'],
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='mean_final_best_cost', ascending=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(by=['algorithm', 'seed'])
    return summary_df, detail_df


def build_detailed_iteration_records(all_results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for algo_name, run_results in all_results.items():
        for r in run_results:
            for rec in r['iteration_records']:
                row = dict(rec)
                row['algorithm'] = algo_name
                row['seed'] = r['seed']
                row['time_limit_seconds'] = r['time_limit_seconds']
                rows.append(row)

    detail_iter_df = pd.DataFrame(rows)
    preferred_cols = [
        'algorithm', 'seed', 'time_limit_seconds', 'iteration', 'elapsed_seconds',
        'best_cost', 'current_cost', 'candidate_cost', 'operator',
        'accepted_move', 'global_improved', 'dp_calls'
    ]
    existing_cols = [c for c in preferred_cols if c in detail_iter_df.columns]
    remaining_cols = [c for c in detail_iter_df.columns if c not in existing_cols]
    return detail_iter_df[existing_cols + remaining_cols].sort_values(
        by=['algorithm', 'seed', 'iteration']
    )


def make_time_ticks(time_limit_seconds: float) -> List[float]:
    if time_limit_seconds <= 10:
        step = 1
    elif time_limit_seconds <= 30:
        step = 5
    elif time_limit_seconds <= 120:
        step = 10
    elif time_limit_seconds <= 300:
        step = 30
    else:
        step = 60

    ticks = []
    value = 0
    while value <= time_limit_seconds + 1e-9:
        ticks.append(value)
        value += step

    if abs(ticks[-1] - time_limit_seconds) > 1e-9:
        ticks.append(time_limit_seconds)

    return ticks


def apply_y_axis_setting(ax, curve_df: pd.DataFrame):
    if PLOT_Y_MIN is not None and PLOT_Y_MAX is not None:
        ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    else:
        y_values = list(curve_df['mean_best_cost'].dropna())
        y_min = min(y_values)
        y_max = max(y_values)
        padding = max((y_max - y_min) * 0.08, 1)
        ax.set_ylim(y_min - padding, y_max + padding)

    if PLOT_Y_TICK_START is not None and PLOT_Y_STEP is not None:
        if PLOT_Y_MAX is not None:
            tick_end = PLOT_Y_MAX
        else:
            tick_end = ax.get_ylim()[1]
        ax.set_yticks(
            [y for y in list(range(int(PLOT_Y_TICK_START), int(tick_end + PLOT_Y_STEP), int(PLOT_Y_STEP)))]
        )


def plot_time_mean_curves(curve_df: pd.DataFrame,
                          time_limit_seconds: float,
                          output_path: str):
    plt.rcParams['font.family'] = 'Times New Roman'

    label_map = {
        "VNS": "VNS-DPH",
        "SA": "SA-DPH",
        "TS": "TS-DPH"
    }

    style_map = {
        "VNS": {
            "color": "#4a7dba",
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 6
        },
        "SA": {
            "color": "#ec6343",
            "marker": "^",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 7
        },
        "TS": {
            "color": "#f5b46f",
            "marker": "D",
            "linestyle": "-",
            "linewidth": 2,
            "markersize": 6
        }
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_order = ["VNS", "SA", "TS"]

    for algo in plot_order:
        sub = curve_df[curve_df["algorithm"] == algo].sort_values("elapsed_seconds")
        if sub.empty:
            continue

        mark_every = max(1, len(sub) // 20)

        ax.plot(
            sub["elapsed_seconds"],
            sub["mean_best_cost"],
            label=label_map.get(algo, algo),
            color=style_map[algo]["color"],
            marker=style_map[algo]["marker"],
            linestyle=style_map[algo]["linestyle"],
            linewidth=style_map[algo]["linewidth"],
            markersize=style_map[algo]["markersize"],
            markevery=mark_every,
            markerfacecolor=style_map[algo]["color"],
            markeredgewidth=0
        )


    ax.set_xlabel("Running time (s)", fontsize=15)
    ax.set_ylabel("Total weighted completion time", fontsize=15)
    ax.set_title("")

    # 左侧略微留白，避免0点与纵轴压在一起
    ax.set_xlim(-0.02 * time_limit_seconds, time_limit_seconds)
    ax.set_xticks(make_time_ticks(time_limit_seconds))

    apply_y_axis_setting(ax, curve_df)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)

    ax.legend(
        loc="upper right",
        fontsize=11,
        frameon=True,
        edgecolor="black"
    )

    ax.tick_params(axis="both", labelsize=13)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ================= 8. 主程序 =================
if __name__ == '__main__':
    base_dir = os.getcwd()

    print('=' * 90)
    print('启动多规模三算法对比实验：VNS（主算法） vs SA / TS')
    print(f'运行时间停止准则: {TIME_LIMIT_SECONDS} 秒/算法/seed')
    print(f'随机种子: {SEEDS}')
    print(f'数据集规模: {DATASET_SIZES}')
    print('数据目录格式: 50_time / 60_time / 70_time / 80_time / 90_time')
    print('输出目录格式: 50_comparison_outputs_time / 60_comparison_outputs_time / ...')
    print('=' * 90)

    algorithm_specs = [
        ('VNS', vns_optimize),
        ('SA', sa_optimize),
        ('TS', ts_optimize),
    ]

    all_scale_summary = []
    all_scale_detail = []
    total_start = time.perf_counter()

    for instance_size in DATASET_SIZES:
        data_dir = os.path.join(base_dir, DATASET_FOLDER_TEMPLATE.format(size=instance_size))
        output_dir = os.path.join(base_dir, OUTPUT_DIR_TEMPLATE.format(size=instance_size))

        print('\n' + '=' * 90)
        print(f'开始处理 {instance_size} 规模数据集')
        print(f'数据目录: {data_dir}')
        print(f'输出目录: {output_dir}')
        print('=' * 90)

        if not initialize_data(data_dir, instance_size=instance_size):
            print(f'[跳过] 未能成功加载 {instance_size} 规模数据集，请检查该文件夹下四个CSV是否齐全。')
            continue

        ensure_output_dir(output_dir)

        all_results = {}
        scale_start = time.perf_counter()

        for algo_name, func in algorithm_specs:
            print(f'\n>>> [{instance_size}] 开始运行 {algo_name} ...')
            run_results = run_algorithm_five_seeds(
                algo_name,
                func,
                SEEDS,
                TIME_LIMIT_SECONDS,
                quiet=True
            )
            all_results[algo_name] = run_results

            mean_final_cost = sum(r['best_cost'] for r in run_results) / len(run_results)
            mean_runtime = sum(r['elapsed_seconds'] for r in run_results) / len(run_results)
            mean_iterations = sum(r['actual_iterations'] for r in run_results) / len(run_results)

            print(
                f'[{instance_size}] {algo_name} 完成 | mean final best cost = {mean_final_cost:.4f} | '
                f'mean runtime = {mean_runtime:.4f}s | '
                f'mean iterations = {mean_iterations:.2f}'
            )

        summary_df, detail_df = build_summary_tables(
            all_results,
            TIME_LIMIT_SECONDS
        )
        summary_df.insert(0, 'instance_size', instance_size)
        summary_df.insert(1, 'data_dir', data_dir)
        detail_df.insert(0, 'instance_size', instance_size)
        detail_df.insert(1, 'data_dir', data_dir)

        curve_df = build_time_based_mean_curve(
            all_results,
            TIME_LIMIT_SECONDS,
            grid_points=TIME_GRID_POINTS
        )
        curve_df.insert(0, 'instance_size', instance_size)
        curve_df.insert(1, 'data_dir', data_dir)

        detailed_iteration_df = build_detailed_iteration_records(all_results)
        detailed_iteration_df.insert(0, 'instance_size', instance_size)
        detailed_iteration_df.insert(1, 'data_dir', data_dir)

        summary_csv_path = os.path.join(output_dir, SUMMARY_CSV)
        detail_csv_path = os.path.join(output_dir, DETAIL_CSV)
        curve_csv_path = os.path.join(output_dir, MEAN_CURVE_CSV)
        detailed_iteration_csv_path = os.path.join(output_dir, DETAILED_ITERATION_CSV)
        plot_path = os.path.join(output_dir, PLOT_FILENAME)

        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        detail_df.to_csv(detail_csv_path, index=False, encoding='utf-8-sig')
        curve_df.to_csv(curve_csv_path, index=False, encoding='utf-8-sig')
        detailed_iteration_df.to_csv(detailed_iteration_csv_path, index=False, encoding='utf-8-sig')

        plot_time_mean_curves(
            curve_df=curve_df,
            time_limit_seconds=TIME_LIMIT_SECONDS,
            output_path=plot_path
        )

        all_scale_summary.append(summary_df)
        all_scale_detail.append(detail_df)

        print('\n' + '-' * 90)
        print(f'{instance_size} 规模实验结束 | 本规模耗时: {time.perf_counter() - scale_start:.4f}s')
        print(f'汇总CSV: {summary_csv_path}')
        print(f'运行明细CSV: {detail_csv_path}')
        print(f'逐迭代明细CSV: {detailed_iteration_csv_path}')
        print(f'时间均值曲线CSV: {curve_csv_path}')
        print(f'对比图: {plot_path}')
        print('\n[本规模汇总结果]')
        print(summary_df.to_string(index=False))
        print('-' * 90)

    if all_scale_summary:
        combined_summary_df = pd.concat(all_scale_summary, ignore_index=True)
        combined_detail_df = pd.concat(all_scale_detail, ignore_index=True)

        combined_summary_path = os.path.join(base_dir, 'all_scales_three_algorithms_summary_time_limited.csv')
        combined_detail_path = os.path.join(base_dir, 'all_scales_three_algorithms_detail_time_limited.csv')

        combined_summary_df.to_csv(combined_summary_path, index=False, encoding='utf-8-sig')
        combined_detail_df.to_csv(combined_detail_path, index=False, encoding='utf-8-sig')

        print('\n' + '=' * 90)
        print('全部规模实验结束')
        print(f'总耗时: {time.perf_counter() - total_start:.4f}s')
        print(f'全部规模汇总CSV: {combined_summary_path}')
        print(f'全部规模运行明细CSV: {combined_detail_path}')
        print('\n[全部规模汇总结果]')
        print(combined_summary_df.to_string(index=False))
        print('=' * 90)
    else:
        print('\n没有任何数据集被成功加载，程序结束。')
