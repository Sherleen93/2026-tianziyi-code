import pandas as pd
import random
import copy
import time
import math
import os
from typing import List, Dict, Set, Any, Optional, Tuple
from itertools import product

# ================= 1. 全局参数设定 =================
PI = 120
SIGMA = 20
A = 10
B = 10
C = 10
BW = 3
K_FILTER = 1
RANDOM_SEED = 4

# ================= 2. 实验总控参数 =================
# 原代码固定的是迭代次数 MAX_ITERATIONS = 250；
# 现在改为固定运行时间，其他公共参数、数据集、seed 和算法逻辑保持不变。
TIME_LIMIT_SECONDS = 300
SEEDS = [3, 22, 45, 42, 66]

OUTPUT_DIR = 'comparison_outputs_time'
SUMMARY_CSV = 'large_scale_fcfs_vns_greedy_vns_dph_summary.csv'
DETAIL_CSV = 'large_scale_fcfs_vns_greedy_vns_dph_detail.csv'

# 时间曲线对齐点数。101 表示从 0 到 TIME_LIMIT_SECONDS 等距取 101 个点。

# ================= 批量数据集设置 =================
# 数据集文件夹需要和本代码放在同一目录下，例如：
# 50_time/orders.csv, 50_time/order_details.csv, 50_time/products.csv, 50_time/inventory.csv
DATASET_SIZES = [300]
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


# ================= VNS 参数（主算法） =================
# 算子保持不变；VNS 外层采用“shaking + local search + neighborhood change”的标准流程。
VNS_RIDE_SHARE_WINDOW = 80
VNS_RIDE_SHARE_REQUIRE = 'either'      # {'sku','rack','either','both'}


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

    # Preserve the original order in orders.csv for the FCFS benchmark.
    fcfs_sequence = [int(x) for x in orders_df['order_id'].tolist()]

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
    return order_info, order_needs, rack_inv, zones, order_demands, fcfs_sequence


# These globals are refreshed before running each instance size.
ORDER_INFO = {}
ORDER_NEEDS_MASTER = {}
RACK_INV = {}
ZONES = []
ORDER_TOTAL_DEMANDS = {}
ORDER_SEQUENCE_DATASET = []
ORDER_SKU_CACHE = {}
ORDER_RACKS_CACHE = {}
CURRENT_DATA_DIR = None
CURRENT_INSTANCE_SIZE = None


def initialize_data(data_dir: str, instance_size: Optional[int] = None) -> bool:
    """Load one dataset and refresh all global caches used by DPH and metaheuristics."""
    global ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS, ORDER_SEQUENCE_DATASET
    global ORDER_SKU_CACHE, ORDER_RACKS_CACHE, CURRENT_DATA_DIR, CURRENT_INSTANCE_SIZE

    data = load_data(data_dir)
    if data[0] is None:
        return False

    ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS, ORDER_SEQUENCE_DATASET = data
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



# ================= 4B. Greedy rack scheduling decoder =================
def evaluate_greedy(sequence: List[int], return_details: bool = False, deadline: Optional[float] = None):
    """
    Greedy rack scheduling decoder used by FCFS-Greedy and VNS-Greedy.

    Given an order sequence, orders are released to the workbench according to
    the same rolling capacity rule as DPH. In each slot and for each zone, the
    decoder selects the rack that can pick the largest number of currently
    required SKUs under the slot picking-time capacity. Ties are broken by:
    1) covering more active orders, 2) covering more required SKUs in total,
    3) avoiding rack switching, and 4) smaller rack id for determinism.

    The completion-time calculation is identical to the DPH evaluator.
    """
    check_time_limit(deadline)

    if not sequence:
        return (0, None) if return_details else 0

    active = list(sequence[:C])
    next_pos = len(active)
    rem = {oid: copy.deepcopy(ORDER_NEEDS_MASTER[oid]) for oid in active}
    last_racks = {z: -1 for z in ZONES}
    completed_slots: Dict[int, int] = {}
    trajectory = []
    slot_count = 0
    total_picked_items = 0
    total_switches = 0

    def no_remaining_orders() -> bool:
        return (not active) and next_pos >= len(sequence)

    while not no_remaining_orders():
        check_time_limit(deadline)
        slot_count += 1
        chosen_racks: Dict[int, Optional[int]] = {}
        chosen_switches: Dict[int, int] = {}

        # Select one rack for each zone greedily.
        for z in ZONES:
            active_needs_z = set()
            for oid in active:
                active_needs_z |= set(rem[oid][z])

            if not active_needs_z:
                chosen_racks[z] = None
                chosen_switches[z] = 0
                continue

            best_rid = None
            best_is_sw = 0
            best_key = None

            for rid, skus in RACK_INV[z].items():
                is_sw = 1 if rid != last_racks[z] else 0
                q_limit = (SIGMA - B * is_sw) // A
                if q_limit <= 0:
                    continue

                overlap_skus = set(skus) & active_needs_z
                overlap_count = len(overlap_skus)
                if overlap_count <= 0:
                    continue

                pickable_count = min(overlap_count, q_limit)
                covered_orders = sum(1 for oid in active if set(rem[oid][z]) & set(skus))

                # Higher is better. -is_sw prefers no switch; -rid makes the rule deterministic.
                key = (pickable_count, covered_orders, overlap_count, -is_sw, -rid)
                if best_key is None or key > best_key:
                    best_key = key
                    best_rid = rid
                    best_is_sw = is_sw

            chosen_racks[z] = best_rid
            chosen_switches[z] = best_is_sw if best_rid is not None else 0

        step_picked = 0
        step_log = {
            'slot': slot_count,
            'active_at_start': list(active),
            'racks': dict(chosen_racks),
            'detailed_skus': {z: [] for z in ZONES},
        }

        # Execute picking for the selected racks.
        for z in ZONES:
            rid = chosen_racks[z]
            if rid is None:
                continue

            is_sw = chosen_switches[z]
            q_limit = (SIGMA - B * is_sw) // A
            last_racks[z] = rid
            total_switches += is_sw

            # Same within-zone picking priority as the DPH evaluator: orders with fewer
            # remaining SKUs in this zone are served first.
            for oid in sorted(active, key=lambda x: len(rem[x][z]) if rem[x][z] else 999):
                if q_limit <= 0:
                    break
                hits = sorted(set(rem[oid][z]) & RACK_INV[z][rid])
                for sku in hits:
                    if q_limit <= 0:
                        break
                    rem[oid][z].remove(sku)
                    q_limit -= 1
                    step_picked += 1
                    total_picked_items += 1
                    if return_details:
                        step_log['detailed_skus'][z].append(f"订单{oid}(SKU:{sku})")

        done_this_slot = [oid for oid in list(active) if all(not rem[oid][zn] for zn in ZONES)]
        for oid in done_this_slot:
            active.remove(oid)
            completed_slots[oid] = slot_count

        while len(active) < C and next_pos < len(sequence):
            new_oid = sequence[next_pos]
            active.append(new_oid)
            rem[new_oid] = copy.deepcopy(ORDER_NEEDS_MASTER[new_oid])
            next_pos += 1

        if return_details:
            trajectory.append(step_log)

        # Safety guard: if no SKU can be picked and no order can be completed, the greedy
        # decoder is stuck, which should not happen for valid datasets with sufficient inventory.
        if step_picked == 0 and not done_this_slot and active:
            if return_details:
                return float('inf'), {
                    'completed_slots': completed_slots,
                    'trajectory': trajectory,
                    'slot_count': slot_count,
                    'last_racks': last_racks,
                    'total_picked_items': total_picked_items,
                    'total_switches': total_switches,
                    'status': 'stuck'
                }
            return float('inf')

    total_cost = sum(
        ORDER_INFO[o]['weight'] * ((completed_slots[o] - 1) * SIGMA + PI)
        for o in sequence
    )

    if return_details:
        final_state = {
            'completed_slots': completed_slots,
            'trajectory': trajectory,
            'slot_count': slot_count,
            'last_racks': last_racks,
            'total_picked_items': total_picked_items,
            'total_switches': total_switches,
            'status': 'ok'
        }
        return total_cost, final_state

    return total_cost

# ================= 5. 共用工具与算子 =================
def build_initial_sequence() -> List[int]:
    return sorted(ORDER_INFO.keys(), key=lambda oid: (-ORDER_INFO[oid]['weight'], -ORDER_TOTAL_DEMANDS[oid]))


def build_fcfs_sequence() -> List[int]:
    """Return the order sequence exactly as provided in orders.csv."""
    return [oid for oid in ORDER_SEQUENCE_DATASET if oid in ORDER_INFO]


def make_cost_evaluator(deadline: Optional[float] = None):
    dp_cache: Dict[Tuple[int, ...], float] = {}

    def get_cost(seq: List[int]) -> float:
        seq_tuple = tuple(seq)
        if seq_tuple not in dp_cache:
            check_time_limit(deadline)
            dp_cache[seq_tuple] = evaluate_dp(list(seq), return_details=False, deadline=deadline)
        return dp_cache[seq_tuple]

    return get_cost, dp_cache


def make_greedy_cost_evaluator(deadline: Optional[float] = None):
    greedy_cache: Dict[Tuple[int, ...], float] = {}

    def get_cost(seq: List[int]) -> float:
        seq_tuple = tuple(seq)
        if seq_tuple not in greedy_cache:
            check_time_limit(deadline)
            greedy_cache[seq_tuple] = evaluate_greedy(list(seq), return_details=False, deadline=deadline)
        return greedy_cache[seq_tuple]

    return get_cost, greedy_cache


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


# ================= 6. 三种算法（FCFS-Greedy / VNS-Greedy / VNS-DPH） =================
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
        'algorithm': 'VNS-DPH',
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




def fcfs_greedy_optimize(time_limit_seconds: float = TIME_LIMIT_SECONDS,
                         quiet: bool = True,
                         random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    FCFS-Greedy:
    1) Orders are processed according to their original order in orders.csv;
    2) No order sequence optimization is performed;
    3) Rack scheduling is generated by the same greedy decoder used in VNS-Greedy;
    4) Only the final objective value is recorded.
    """
    start_time = time.perf_counter()
    deadline = start_time + time_limit_seconds

    seq = build_fcfs_sequence()
    try:
        best_cost = evaluate_greedy(seq, return_details=False, deadline=deadline)
    except TimeLimitReached:
        best_cost = float('inf')

    total_elapsed = time.perf_counter() - start_time

    if not quiet:
        print(f"[FCFS-Greedy] final cost = {best_cost} | runtime = {total_elapsed:.4f}s")

    return {
        'algorithm': 'FCFS-Greedy',
        'seed': random_seed,
        'initial_cost': best_cost,
        'best_cost': best_cost,
        'best_seq': seq,
        'elapsed_seconds': total_elapsed,
        'actual_iterations': 0,
        'time_limit_seconds': time_limit_seconds,
        'dp_cache_size': 0,
        'used_params': {
            'order_sequence': 'FCFS sequence from orders.csv',
            'rack_decoder': 'greedy rack scheduling',
            'stopping_criterion': 'single deterministic decoding run'
        }
    }

def vns_greedy_optimize(time_limit_seconds: float = TIME_LIMIT_SECONDS,
                 quiet: bool = True,
                 random_seed: int = RANDOM_SEED) -> Dict[str, Any]:
    """
    VNS-Greedy（运行时间停止准则）：
    1) 外层 VNS 搜索订单投产序列；
    2) 内层 evaluate_greedy() 作为 greedy 评价器；
    3) 算子保持不变：N1 ride-sharing, N2 shift, N3 reverse；
    4) 不再设置单独的 SHAKE 状态，也不区分局部/大范围参数；
    5) 接受准则：仅接受改进解。
    """
    start_time = time.perf_counter()
    deadline = start_time + time_limit_seconds
    rng = random.Random(random_seed)
    n_orders = len(ORDER_INFO)
    get_cost, dp_cache = make_greedy_cost_evaluator(deadline=deadline)
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
                    f"[VNS-Greedy][Seed {random_seed}][Iter {iteration:04d}] "
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
        'algorithm': 'VNS-Greedy',
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




# ================= 7. 多次运行、均值与CSV输出 =================
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_algorithm_five_seeds(algorithm_name: str, optimize_func, seeds: List[int],
                             time_limit_seconds: float, quiet: bool = True):
    """Run FCFS-Greedy once and VNS-based algorithms for all specified seeds."""
    if algorithm_name == 'FCFS-Greedy':
        return [optimize_func(time_limit_seconds=time_limit_seconds, quiet=quiet, random_seed=seeds[0])]

    results = []
    for seed in seeds:
        res = optimize_func(time_limit_seconds=time_limit_seconds, quiet=quiet, random_seed=seed)
        results.append(res)
    return results



def build_summary_tables(all_results: Dict[str, List[Dict[str, Any]]],
                         time_limit_seconds: float):
    """Build final-value tables only; no convergence curves are generated."""
    summary_rows = []
    detail_rows = []

    for algo_name, run_results in all_results.items():
        n_runs = len(run_results)
        mean_initial_cost = sum(r['initial_cost'] for r in run_results) / n_runs
        mean_final_cost = sum(r['best_cost'] for r in run_results) / n_runs
        mean_runtime = sum(r['elapsed_seconds'] for r in run_results) / n_runs
        mean_iterations = sum(r.get('actual_iterations', 0) for r in run_results) / n_runs

        best_run = min(run_results, key=lambda r: r['best_cost'])
        worst_run = max(run_results, key=lambda r: r['best_cost'])

        summary_rows.append({
            'algorithm': algo_name,
            'time_limit_seconds': time_limit_seconds,
            'n_runs': n_runs,
            'mean_initial_cost': mean_initial_cost,
            'mean_final_best_cost': mean_final_cost,
            'best_seed': best_run['seed'],
            'best_final_cost': best_run['best_cost'],
            'worst_seed': worst_run['seed'],
            'worst_final_cost': worst_run['best_cost'],
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
                'actual_iterations': r.get('actual_iterations', 0),
                'time_limit_seconds': time_limit_seconds,
                'cache_size': r.get('dp_cache_size', 0),
            })

    summary_df = pd.DataFrame(summary_rows).sort_values(by='mean_final_best_cost', ascending=True)
    detail_df = pd.DataFrame(detail_rows).sort_values(by=['algorithm', 'seed'])
    return summary_df, detail_df


# ================= 8. 主程序 =================
if __name__ == '__main__':
    base_dir = os.getcwd()

    print('=' * 90)
    print('启动多规模大规模对比实验：FCFS-Greedy vs VNS-Greedy vs VNS-DPH')
    print(f'VNS类算法VNS类算法运行时间停止准则: {TIME_LIMIT_SECONDS} 秒/算法/seed')
    print(f'VNSVNS随机种子: {SEEDS}')
    print(f'数据集规模: {DATASET_SIZES}')
    print('数据目录格式: 50_time / 60_time / 70_time / 80_time / 90_time')
    print('输出目录格式: 50_comparison_outputs_time / 60_comparison_outputs_time / ...')
    print('本版本不生成收敛图，仅记录最终目标函数值。')
    print('=' * 90)

    algorithm_specs = [
        ('FCFS-Greedy', fcfs_greedy_optimize),
        #('VNS-Greedy', vns_greedy_optimize),
        #('VNS-DPH', vns_optimize),
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
            mean_iterations = sum(r.get('actual_iterations', 0) for r in run_results) / len(run_results)

            print(
                f'[{instance_size}] {algo_name} 完成 | mean final cost = {mean_final_cost:.4f} | '
                f'mean runtime = {mean_runtime:.4f}s | '
                f'mean iterations = {mean_iterations:.2f} | n_runs = {len(run_results)}'
            )

        summary_df, detail_df = build_summary_tables(
            all_results,
            TIME_LIMIT_SECONDS
        )
        summary_df.insert(0, 'instance_size', instance_size)
        summary_df.insert(1, 'data_dir', data_dir)
        detail_df.insert(0, 'instance_size', instance_size)
        detail_df.insert(1, 'data_dir', data_dir)

        summary_csv_path = os.path.join(output_dir, 'large_scale_fcfs_vns_greedy_vns_dph_summary.csv')
        detail_csv_path = os.path.join(output_dir, 'large_scale_fcfs_vns_greedy_vns_dph_detail.csv')

        summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        detail_df.to_csv(detail_csv_path, index=False, encoding='utf-8-sig')

        all_scale_summary.append(summary_df)
        all_scale_detail.append(detail_df)

        print('\n' + '-' * 90)
        print(f'{instance_size} 规模实验结束 | 本规模耗时: {time.perf_counter() - scale_start:.4f}s')
        print(f'汇总CSV: {summary_csv_path}')
        print(f'运行明细CSV: {detail_csv_path}')
        print('\n[本规模汇总结果]')
        print(summary_df.to_string(index=False))
        print('-' * 90)

    if all_scale_summary:
        combined_summary_df = pd.concat(all_scale_summary, ignore_index=True)
        combined_detail_df = pd.concat(all_scale_detail, ignore_index=True)

        combined_summary_path = os.path.join(base_dir, 'all_scales_large_scale_fcfs_vns_greedy_vns_dph_summary.csv')
        combined_detail_path = os.path.join(base_dir, 'all_scales_large_scale_fcfs_vns_greedy_vns_dph_detail.csv')

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
