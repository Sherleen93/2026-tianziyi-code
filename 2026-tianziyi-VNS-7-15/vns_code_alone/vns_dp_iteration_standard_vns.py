import pandas as pd
import random
import copy
import time
import math
from typing import List, Dict, Set, Any, Optional, Tuple
from itertools import product

# ================= 1. 全局参数设定 =================
PI = 120
SIGMA = 20
A = 10
B = 10
C = 3
BW = 5
K_FILTER = 4
RANDOM_SEED = 3

# ================= VNS 参数集中 =================
# 算子保持不变；外层改为更标准的 VNS 流程：
# Nk shaking -> N1/N2/N3 local search -> neighborhood change。
# 停止准则仍采用固定外层迭代次数。
VNS_RIDE_SHARE_WINDOW = 6             # 判断“搭顺风车”时参考目标位置前方连续窗口大小；建议保持为1
VNS_RIDE_SHARE_REQUIRE = 'either'      # {'sku','rack','either','both'}

# ================= 2. 数据加载与哈希预计算 =================
def load_data():
    try:
        orders_df = pd.read_csv('orders.csv')
        details_df = pd.read_csv('order_details.csv')
        products_df = pd.read_csv('products.csv')
        inventory_df = pd.read_csv('inventory.csv')
    except FileNotFoundError:
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


DATA = load_data()
ORDER_SKU_CACHE = {}
ORDER_RACK_SKU_CACHE = {}
ORDER_RACKS_CACHE = {}  # 缓存每个订单依赖的物理货架 (Zone_ID, Rack_ID)

if DATA[0] is not None:
    ORDER_INFO, ORDER_NEEDS_MASTER, RACK_INV, ZONES, ORDER_TOTAL_DEMANDS = DATA
    for oid, zones_dict in ORDER_NEEDS_MASTER.items():
        ORDER_SKU_CACHE[oid] = set().union(*zones_dict.values())

        racks = set()
        for z, skus in zones_dict.items():
            if not skus:
                continue
            for sku in skus:
                for rid, rack_skus in RACK_INV[z].items():
                    if sku in rack_skus:
                        racks.add((z, rid))
        ORDER_RACKS_CACHE[oid] = racks


# ================= 3. 底层 DP (原逻辑完全保留) =================
def evaluate_dp(sequence: List[int], return_details: bool = False):
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
        if all(not st['active'] and st['next_pos'] >= len(sequence) for st in frontier):
            def get_objective(x):
                return sum(ORDER_INFO[o]['weight'] * ((x['completed_slots'][o] - 1) * SIGMA + PI) for o in sequence)

            best_final = min(frontier, key=get_objective)
            break

        next_candidates = []
        for st in frontier:
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
                                    key=lambda x: len(rem_temp_eval[x][z]) if rem_temp_eval[x][z] else 999):
                        if q_limit <= 0:
                            break
                        hits = set(rem_temp_eval[o][z]) & RACK_INV[z][action[i][0]]
                        for sku in list(hits):
                            if q_limit <= 0:
                                break
                            rem_temp_eval[o][z].remove(sku)
                            q_limit -= 1

            active_after_action = [o for o in st['active'] if any(rem_temp_eval[o][zn] for zn in ZONES)]
            min_rem_single = min((sum(len(rem_temp_eval[o][z]) for z in ZONES) for o in active_after_action), default=0)
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


# ================= 4. 外层 VNS-DPH（固定迭代次数 + 标准VNS流程） =================

def vns_optimize(
    max_iterations: int = 100,
    quiet: bool = False,
    random_seed: int = RANDOM_SEED,
    vns_ride_share_window: int = VNS_RIDE_SHARE_WINDOW,
    vns_ride_share_require: str = VNS_RIDE_SHARE_REQUIRE,
):
    """
    VNS-DPH（固定迭代次数停止准则）：
    1) 外层 VNS 搜索订单投产序列；
    2) 内层 evaluate_dp() 作为 DPH 评价器；
    3) 算子保持不变：N1 ride-sharing forward insertion, N2 shift, N3 reverse；
    4) 不再设置单独的 SHAKE 状态，也不区分局部/大范围参数；
    5) 每个外层迭代执行：Nk shaking -> N1/N2/N3 local search -> neighborhood change；
    6) 接受准则：仅接受改进解。
    """
    start_vns_time = time.time()
    rng = random.Random(random_seed)
    n_orders = len(ORDER_INFO)
    K_MAX = 3

    initial_seq = sorted(
        ORDER_INFO.keys(),
        key=lambda oid: (-ORDER_INFO[oid]['weight'], -ORDER_TOTAL_DEMANDS[oid])
    )

    dp_cache: Dict[Tuple[int, ...], float] = {}
    dph_evaluations = 0

    # -------- helpers --------
    def elapsed() -> float:
        return time.time() - start_vns_time

    def get_cost(seq: List[int]) -> float:
        nonlocal dph_evaluations
        seq_tuple = tuple(seq)
        if seq_tuple not in dp_cache:
            dp_cache[seq_tuple] = evaluate_dp(list(seq), return_details=False)
            dph_evaluations += 1
        return dp_cache[seq_tuple]

    def has_shared_structure(candidate_oid: int, neighbor_oids: List[int]) -> bool:
        if not neighbor_oids:
            return False

        neighbor_skus = set()
        neighbor_racks = set()
        for oid in neighbor_oids:
            neighbor_skus |= ORDER_SKU_CACHE[oid]
            neighbor_racks |= ORDER_RACKS_CACHE[oid]

        sku_shared = len(ORDER_SKU_CACHE[candidate_oid] & neighbor_skus) > 0
        rack_shared = len(ORDER_RACKS_CACHE[candidate_oid] & neighbor_racks) > 0

        if vns_ride_share_require == 'sku':
            return sku_shared
        if vns_ride_share_require == 'rack':
            return rack_shared
        if vns_ride_share_require == 'both':
            return sku_shared and rack_shared
        return sku_shared or rack_shared

    # -------- N1: ride-sharing forward insertion --------
    def n1_ride_sharing_forward_insertion(
        seq: List[int],
        label: str = 'N1_ride_sharing_forward_insertion'
    ) -> Tuple[List[int], Dict[str, Any]]:
        s = seq[:]
        n = len(s)
        if n < 2:
            return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

        i = rng.randrange(1, n)
        candidate_oid = s[i]
        matched_positions = []
        window_size = max(1, int(vns_ride_share_window))

        # 只保留真正能产生前插的位置，避免插回原位置导致无效移动。
        for j in range(i - 2, -1, -1):
            left = max(0, j - window_size + 1)
            window_oids = s[left:j + 1]
            if has_shared_structure(candidate_oid, window_oids):
                matched_positions.append(j)

        if not matched_positions:
            return s, {
                'operator': label,
                'move_applied': False,
                'reason': 'no_match',
                'from_index': i
            }

        # 不再区分 local 和 shake 的强度；统一在所有可搭顺风车位置中随机选择一个。
        matched_j = rng.choice(matched_positions)
        item = s.pop(i)
        s.insert(matched_j + 1, item)

        return s, {
            'operator': label,
            'move_applied': s != seq,
            'from_index': i,
            'to_after_index': matched_j,
            'reason': 'matched',
            'matched_count': len(matched_positions),
        }

    # -------- N2: shift --------
    def n2_shift(seq: List[int], label: str = 'N2_shift') -> Tuple[List[int], Dict[str, Any]]:
        s = seq[:]
        n = len(s)
        if n < 3:
            return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

        # 不再区分局部/大范围；直接在合法范围内随机选择连续块。
        block_len = rng.randint(1, n - 1)
        a = rng.randint(0, n - block_len)
        b = a + block_len - 1

        # c 表示插入到原序列中某个位置之后；c=-1 表示插到最前面。
        valid_c = list(range(-1, a)) + list(range(b + 1, n))
        if not valid_c:
            return s, {'operator': label, 'move_applied': False, 'reason': 'no_valid_target'}

        c = rng.choice(valid_c)
        block = s[a:b + 1]
        remaining = s[:a] + s[b + 1:]

        if c < a:
            insert_pos = c + 1
        else:
            insert_pos = c - block_len + 1

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

    # -------- N3: reverse --------
    def n3_reverse(seq: List[int], label: str = 'N3_reverse') -> Tuple[List[int], Dict[str, Any]]:
        s = seq[:]
        n = len(s)
        if n < 2:
            return s, {'operator': label, 'move_applied': False, 'reason': 'too_short'}

        # 不再区分局部/大范围；直接在合法范围内随机选择反转长度。
        rev_len = rng.randint(2, n)
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

    def apply_neighborhood(
        seq: List[int],
        k: int,
        phase: str
    ) -> Tuple[List[int], Dict[str, Any]]:
        if k == 1:
            cand_seq, meta = n1_ride_sharing_forward_insertion(seq, label=f'{phase}_N1_ride_sharing')
        elif k == 2:
            cand_seq, meta = n2_shift(seq, label=f'{phase}_N2_shift')
        elif k == 3:
            cand_seq, meta = n3_reverse(seq, label=f'{phase}_N3_reverse')
        else:
            raise ValueError(f'Invalid neighborhood index: {k}')

        meta['phase'] = phase
        meta['neighborhood_k'] = k
        return cand_seq, meta

    def local_search(
        seq: List[int],
        cost: float
    ) -> Tuple[List[int], float, int, List[str]]:
        """VND-style local search using N1, N2, and N3."""
        local_seq = seq[:]
        local_cost = cost
        st = 1
        local_steps = 0
        local_ops = []

        while st <= K_MAX:
            cand_seq, meta = apply_neighborhood(local_seq, st, phase='local')
            cand_cost = get_cost(cand_seq)
            local_steps += 1
            local_ops.append(meta.get('operator', ''))

            if cand_cost < local_cost:
                local_seq = cand_seq[:]
                local_cost = cand_cost
                st = 1
            else:
                st += 1

        return local_seq, local_cost, local_steps, local_ops

    def state_name(k_value: int) -> str:
        if 1 <= k_value <= K_MAX:
            return f'N{k_value}'
        return 'restart'

    # -------- initialization --------
    initial_cost = get_cost(initial_seq)
    best_seq = initial_seq[:]
    best_cost = initial_cost
    current_seq = initial_seq[:]
    current_cost = initial_cost

    cost_history = [best_cost]
    best_update_iterations = [0]

    iteration_records = [{
        'iteration': 0,
        'best_cost': best_cost,
        'current_cost': current_cost,
        'candidate_cost': current_cost,
        'shaken_cost': current_cost,
        'dp_calls': 1,
        'dph_evaluations_total': dph_evaluations,
        'dp_cache_size': len(dp_cache),
        'global_improved': 0,
        'accepted_move': 0,
        'elapsed_seconds': elapsed(),
        'state_before': 'initial',
        'state_after': 'N1',
        'operator': 'initial',
        'local_search_steps': 0,
        'local_operators': '',
        'best_update': 0,
    }]

    # -------- main VNS loop with fixed outer iterations --------
    iteration = 0
    while iteration < max_iterations:
        k = 1

        while k <= K_MAX and iteration < max_iterations:
            iteration += 1
            state_before = state_name(k)
            cache_size_before = len(dp_cache)

            # Step 1: shaking by Nk
            shaken_seq, shake_meta = apply_neighborhood(current_seq, k, phase='shake')
            shaken_cost = get_cost(shaken_seq)

            # Step 2: local search from shaken sequence
            local_seq, local_cost, local_steps, local_ops = local_search(shaken_seq, shaken_cost)

            accepted_move = 0
            global_improved = 0
            best_update = 0

            # Step 3: neighborhood change
            if local_cost < current_cost:
                current_seq = local_seq[:]
                current_cost = local_cost
                accepted_move = 1

                if current_cost < best_cost:
                    best_seq = current_seq[:]
                    best_cost = current_cost
                    global_improved = 1
                    best_update = 1
                    best_update_iterations.append(iteration)

                k = 1
            else:
                k += 1

            state_after = state_name(k)
            new_dp_calls = len(dp_cache) - cache_size_before

            if not quiet:
                print(
                    f"[Iter {iteration:04d}] k={state_before} | "
                    f"shake={shake_meta.get('operator', '')} | "
                    f"shaken={shaken_cost} | local={local_cost} | "
                    f"当前值={current_cost} | 全局最优={best_cost} | "
                    f"接受={accepted_move} | 状态: {state_before} -> {state_after} | "
                    f"local_steps={local_steps} | 新DP调用={new_dp_calls} | 缓存={len(dp_cache)}"
                )

            cost_history.append(best_cost)
            iteration_records.append({
                'iteration': iteration,
                'best_cost': best_cost,
                'current_cost': current_cost,
                'candidate_cost': local_cost,
                'shaken_cost': shaken_cost,
                'dp_calls': new_dp_calls,
                'dph_evaluations_total': dph_evaluations,
                'dp_cache_size': len(dp_cache),
                'global_improved': global_improved,
                'accepted_move': accepted_move,
                'elapsed_seconds': elapsed(),
                'state_before': state_before,
                'state_after': state_after,
                'operator': shake_meta.get('operator', ''),
                'local_search_steps': local_steps,
                'local_operators': ' + '.join(local_ops),
                'best_update': best_update,
                'meta': str(shake_meta),
            })

    last_improvement_iteration = best_update_iterations[-1] if best_update_iterations else 0
    return {
        'initial_cost': initial_cost,
        'best_seq': best_seq,
        'best_cost': best_cost,
        'cost_history': cost_history,
        'iteration_records': iteration_records,
        'best_update_iterations': best_update_iterations,
        'last_improvement_iteration': last_improvement_iteration,
        'actual_iterations': len(iteration_records) - 1,
        'elapsed_seconds': elapsed(),
        'dph_evaluations': dph_evaluations,
        'used_params': {
            'max_iterations': max_iterations,
            'random_seed': random_seed,
            'vns_ride_share_window': vns_ride_share_window,
            'vns_ride_share_require': vns_ride_share_require,
            'neighborhoods': 'N1 ride-sharing forward insertion; N2 shift; N3 reverse',
            'stopping_criterion': 'fixed outer neighborhood iterations',
            'acceptance_rule': 'only improving solutions are accepted',
            'vns_scheme': 'Nk shaking -> VND-style local search using N1/N2/N3 -> neighborhood change',
            'separate_shake_state': False,
            'separate_local_large_scale_parameters': False,
        }
    }


# ================= 5. 保存逐迭代记录 =================
def save_iteration_records(iteration_records: List[Dict[str, Any]], output_path: str = 'standard_vns_iteration_history.csv'):
    pd.DataFrame(iteration_records).to_csv(output_path, index=False, encoding='utf-8-sig')


# ================= 6. 主程序执行 =================
if __name__ == "__main__":
    if DATA[0] is None:
        exit()

    start_time = time.time()

    # 固定外层迭代次数停止
    MAX_ITERATIONS = 30
    RANDOM_SEED_MAIN = RANDOM_SEED

    print("========== 启动 VNS-DPH 集成调度优化（DP原逻辑保留） ==========")
    print("目标：以固定外层迭代次数为停止准则，记录‘目标函数-迭代次数’变化")
    print("外层：标准VNS流程，即 Nk shaking -> N1/N2/N3 局部搜索 -> neighborhood change")
    print("邻域结构：N1顺风车前插、N2 shift、N3 reverse；不再设置单独SHAKE状态，也不区分局部/大范围参数")
    print("接受准则：仅接受改进解；若改进则回到N1，否则切换到下一个邻域")
    print(
        f"参数: max_iterations={MAX_ITERATIONS}, "
        f"window={VNS_RIDE_SHARE_WINDOW}, "
        f"ride_share_require={VNS_RIDE_SHARE_REQUIRE}, "
        f"random_seed={RANDOM_SEED_MAIN}"
    )

    results = vns_optimize(
        max_iterations=MAX_ITERATIONS,
        quiet=False,
        random_seed=RANDOM_SEED_MAIN,
        vns_ride_share_window=VNS_RIDE_SHARE_WINDOW,
        vns_ride_share_require=VNS_RIDE_SHARE_REQUIRE,
    )

    initial_cost = results['initial_cost']
    final_sequence = results['best_seq']
    final_cost = results['best_cost']
    iteration_records = results['iteration_records']
    actual_iterations = results['actual_iterations']
    last_improvement_iteration = results['last_improvement_iteration']

    print("\n========== VNS-DPH 外层优化完成 ==========")
    print(f"初始序列加权耗时: {initial_cost}")
    print(f"最终优化加权耗时: {final_cost}")
    print(f"最优订单投产序列: {final_sequence}")
    print(f"实际完成外层迭代次数: {actual_iterations}")
    print(f"DPH 实际评价次数: {results['dph_evaluations']}")
    print(f"最后一次全局改进发生在迭代: {last_improvement_iteration}")
    if actual_iterations > last_improvement_iteration:
        print(f"从迭代 {last_improvement_iteration} 到 {actual_iterations} 未再出现全局改进，可据此观察收敛趋势。")

    csv_path = 'standard_vns_iteration_history.csv'
    save_iteration_records(iteration_records, csv_path)
    print(f"逐迭代记录已保存: {csv_path}")

    print("\n========== 开始生成最终最优序列的物理调度日志 ==========")
    _, best_state = evaluate_dp(final_sequence, return_details=True)

    print("\n" + "#" * 75)
    print("🏆 全局最优解执行追踪 🏆")
    print("#" * 75)

    print("\n[订单完工报告]")
    for oid in final_sequence:
        slot = best_state['completed_slots'][oid]
        time_cost = (slot - 1) * SIGMA + PI
        weight = ORDER_INFO[oid]['weight']
        print(
            f"  > 订单 {oid}: 完工于 Slot {slot} (物理耗时 {time_cost}s) | 权重: {weight} | 加权得分: {weight * time_cost}"
        )

    print("\n" + "=" * 75)
    print("📦 机械臂调度明细")
    print("=" * 75)

    for step in best_state['trajectory']:
        print(f"\n▶ Slot {step['slot']} | 开始时分配至打包台的订单: {step['active_at_start']}")
        for z in ZONES:
            rid = step['racks'][z]
            skus_desc = ", ".join(step['detailed_skus'][z])
            print(
                f"  📍 工作站 {z} -> 停靠货架: {rid if rid else '空闲'} "
                f"| 动作明细: [{skus_desc if skus_desc else '无拣选'}]"
            )

    end_time = time.time()
    run_time = end_time - start_time

    print("\n" + "★" * 75)
    print("✨ 实验执行数据汇总 ✨")
    print(f" 🎯 算法总运行耗时: {run_time:.2f} 秒")
    print("\n[参数设置]")
    for k, v in results['used_params'].items():
        print(f"  {k} = {v}")

    print("\n[前10轮/末10轮迭代摘要]")
    display_records = iteration_records[1:11]
    if actual_iterations > 20:
        display_records += iteration_records[-10:]
    elif actual_iterations > 10:
        display_records = iteration_records[1:]

    seen = set()
    for item in display_records:
        if item['iteration'] in seen:
            continue
        seen.add(item['iteration'])
        print(
            f"  Iter {item['iteration']:04d} | 状态={item['state_before']} -> {item['state_after']} | "
            f"ShakeOp={item['operator']} | Shaken={item.get('shaken_cost', '')} | "
            f"Local={item['candidate_cost']} | Best={item['best_cost']} | "
            f"接受={item['accepted_move']} | 全局更新={item['global_improved']} | "
            f"LocalSteps={item.get('local_search_steps', 0)} | 新DP调用={item.get('dp_calls', 0)} | "
            f"耗时={item['elapsed_seconds']:.1f}s"
        )
    print("★" * 75 + "\n")
