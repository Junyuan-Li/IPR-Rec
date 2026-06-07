import argparse
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np

from llm.llm_client import validate_llm_configuration
from models.core import ONeRecBaseline
from models.native_baseline import NativeONeRecBaseline
from utils.runtime_utils import (
    GLOBAL_SEED,
    configure_stdout_utf8,
    convert_to_serializable,
    install_required_packages,
    print_gpu_info,
    resolve_data_dir,
    set_global_seed,
    validate_data_files,
)


BASELINE_METHOD_NAME = 'Original Backbone'
IDEA_METHOD_NAME = 'Main Idea (Path Planning)'
LLM_METHOD_NAME = 'Ours (LLM + Step3)'
LLM_NO_GUARD_METHOD_NAME = 'Ours w/o Guard'
LLM_NO_REFLECTION_METHOD_NAME = 'Ours w/o Reflection'
LLM_NO_RESCUE_METHOD_NAME = 'Ours w/o Rescue'


DEFAULT_EXP1_CONFIG = {
    'EXP1_NAME': 'EXP1_sanity_check_baseline_validation',
    'EXP1_PROMPT_VERSION': 'exp1_target_oriented_soft_bias_v2',
    'LOCAL_SMOKE_TEST': False,
    'RUN_BASELINE': True,
    'RUN_GREEDY_PATH': True,
    'RUN_LLM_PATH': True,
    'RUN_INTERNAL_ABLATIONS': False,
    'RUN_LLM_NO_GUARD': False,
    'RUN_LLM_NO_REFLECTION': False,
    'RUN_LLM_NO_RESCUE': False,
    'ENABLE_STEP3': True,
    'EXP1_REDUCED_LOGGING': True,
    'EXP1_PARALLEL_EXECUTION': True,
    'SAMPLE_SIZE': 300,
    'RANDOM_SEED': 42,
    'REACH_MIN_SIM': 0.05,
    'IMPRESSION_SIZE': 120,
    'K_LIST': [5, 6],
    'MAX_ROUNDS': 6,
    'DEBUG_USER_LIMIT': 2,
    'LAMBDA0': 0.1,
    'LAMBDA1': 0.2,
    'GRAPH_K': 10,
    'MAX_PATH_LENGTH': 5,
    'LLM_HYBRID_ROUNDS': 2,
    'LLM_CANDIDATE_K': 20,
    'LLM_PATH_K': 5,
    'LLM_TARGET_PRIOR': 0.65,
    'LLM_SAFE_MARGIN': 0.015,
    'LLM_RELEASE_RANK': 3,
    'ENABLE_GATING': True,
    'ENABLE_REPLANNING': True,
    'ENABLE_REFLECTION': True,
}


def _signature(items: List[str]) -> str:
    return '||'.join(str(item) for item in items)


def build_eval_sessions(sample_size: int, data_loader, news_embeds, impression_size: int, random_seed: int) -> List[dict]:
    all_user_ids = sorted(data_loader.users.keys())
    if not all_user_ids:
        return []
    sample_count = min(sample_size, len(all_user_ids))
    session_rng = np.random.default_rng(random_seed)
    sampled_indices = np.sort(session_rng.choice(len(all_user_ids), size=sample_count, replace=False))
    sessions = []
    for user_idx in sampled_indices:
        user_id = all_user_ids[int(user_idx)]
        user_data = data_loader.users[user_id]
        user_history = user_data['history']
        all_impressions = [item_id for item_id in user_data['impressions'] if item_id in news_embeds]
        impressions = data_loader.build_click_biased_impressions(
            user_id,
            all_impressions if all_impressions else list(news_embeds.keys()),
            size=impression_size,
        )
        if len(user_history) < 3 or len(impressions) < 5:
            continue
        sessions.append({
            'user_id': user_id,
            'history': user_history,
            'history_signature': _signature(user_history),
            'impressions': impressions,
            'impression_signature': _signature(sorted(impressions)),
            'candidate_size': len(impressions),
        })
    return sessions


def run_experiment(planner_mode: str, exp_name: str, eval_sessions: List[dict], config: dict, news_embeds, data_loader, overrides: dict | None = None):
    print(f'运行实验: {exp_name}')
    print(f'    planner_mode: {planner_mode}')
    if overrides:
        print(f'    overrides: {overrides}')
    exp_start_time = time.time()
    results = {
        'hit_rate': 0,
        'avg_rounds': 0,
        'round_distribution': defaultdict(int),
        'alpha_distribution': [],
        'initial_target_ranks': [],
        'final_target_ranks': [],
        'best_target_ranks': [],
        'llm_stats': {
            'called': 0,
            'api_success': 0,
            'api_failure': 0,
            'api_error_messages': [],
            'invalid_path': 0,
            'success': 0,
            'fallback': 0,
            'remote_adopted': 0,
            'guard_adopted': 0,
            'rescue_adopted': 0,
            'remote_adopted_rate': 0.0,
            'guard_adopted_rate': 0.0,
            'rescue_adopted_rate': 0.0,
            'avg_path_length': 0.0,
            'success_rate': 0.0,
            'api_success_rate': 0.0,
        },
        'hr_at_k': {k: 0 for k in config['K_LIST']},
        'ioi_at_k': {k: 0.0 for k in config['K_LIST']},
        'ior_at_k': {k: 0.0 for k in config['K_LIST']},
        'trajectories': [],
        'acceptance_rate': 0,
        'feedback_stats': {'accept': 0, 'reject': 0},
        'ratings_logs': [],
        'ranking_logs': [],
        'click_logs': [],
        'success_rounds': [],
        'path_stats': {'total_paths': 0, 'avg_path_length': 0, 'path_success_rate': 0},
        'path_rationality': [],
        'per_user_records': [],
    }
    valid_users = 0
    progress_interval = max(1, len(eval_sessions) // 20) if eval_sessions else 1

    def evaluate_session(payload):
        user_idx, session = payload
        system_cls = NativeONeRecBaseline if planner_mode == 'baseline' else ONeRecBaseline
        common_kwargs = {
            'reach_min_sim': config['REACH_MIN_SIM'],
            'feedback_threshold': 0.5,
            'planner_mode': planner_mode,
            'graph_k': config['GRAPH_K'],
            'max_path_length': config['MAX_PATH_LENGTH'],
            'llm_candidate_k': config['LLM_CANDIDATE_K'],
            'llm_path_k': config['LLM_PATH_K'],
            'llm_target_prior': config['LLM_TARGET_PRIOR'],
            'llm_hybrid_rounds': config['LLM_HYBRID_ROUNDS'],
            'enable_gating': config['ENABLE_GATING'],
            'enable_replanning': config['ENABLE_REPLANNING'],
            'enable_reflection': config['ENABLE_REFLECTION'],
            'debug': ((not config['EXP1_REDUCED_LOGGING']) and (user_idx < config['DEBUG_USER_LIMIT'])),
        }
        if planner_mode != 'baseline':
            common_kwargs['llm_safe_margin'] = config['LLM_SAFE_MARGIN']
            common_kwargs['llm_release_rank'] = config['LLM_RELEASE_RANK']
            common_kwargs['enable_llm_guard'] = True
            common_kwargs['enable_llm_rescue'] = True
        if overrides:
            common_kwargs.update(overrides)
        system = system_cls(
            news_embeds,
            data_loader.news,
            data_loader.news_titles,
            session['history'],
            session['impressions'],
            **common_kwargs,
        )
        system.lambda0 = config['LAMBDA0']
        system.lambda1 = config['LAMBDA1']
        system.alpha = system._compute_openness()
        trajectory = system.run(max_rounds=config['MAX_ROUNDS'])
        return {
            'user_idx': user_idx,
            'user_id': session['user_id'],
            'history_signature': session['history_signature'],
            'impression_signature': session['impression_signature'],
            'candidate_size': session['candidate_size'],
            'system': system,
            'trajectory': trajectory,
        }

    session_payloads = list(enumerate(eval_sessions))
    max_workers = max(1, min(8, os.cpu_count() or 1))
    if config['EXP1_PARALLEL_EXECUTION'] and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            evaluated_sessions = list(executor.map(evaluate_session, session_payloads))
    else:
        evaluated_sessions = [evaluate_session(payload) for payload in session_payloads]

    for processed_count, evaluated in enumerate(evaluated_sessions, 1):
        user_id = evaluated['user_id']
        system = evaluated['system']
        trajectory = evaluated['trajectory']
        valid_users += 1
        if processed_count % progress_interval == 0 or processed_count == len(evaluated_sessions):
            elapsed = time.time() - exp_start_time
            avg_per_user = elapsed / max(processed_count, 1)
            print(f'  progress {processed_count}/{len(evaluated_sessions)} users elapsed={elapsed:.1f}s avg={avg_per_user:.2f}s/user valid={valid_users}')
        if planner_mode == 'llm_path':
            results['llm_stats']['called'] += system.llm_called
            results['llm_stats']['api_success'] += system.llm_api_success
            results['llm_stats']['api_failure'] += system.llm_api_failure
            for error_message in system.llm_api_error_messages:
                if len(results['llm_stats']['api_error_messages']) >= 20:
                    break
                results['llm_stats']['api_error_messages'].append(error_message)
            results['llm_stats']['invalid_path'] += system.llm_invalid_path
            results['llm_stats']['success'] += system.llm_success
            results['llm_stats']['fallback'] += system.llm_fallback
            results['llm_stats']['remote_adopted'] += system.llm_remote_adopted
            results['llm_stats']['guard_adopted'] += system.llm_guard_adopted
            results['llm_stats']['rescue_adopted'] += system.llm_rescue_adopted
            results['llm_stats']['avg_path_length'] += system.llm_path_length_sum
        if len(trajectory) == 0:
            continue

        success_round = next((step['round'] for step in trajectory if step['success']), None)
        hit = success_round is not None
        num_rounds = success_round if hit else len(trajectory)
        target_id = trajectory[0]['target']
        initial_target_rating = system._target_similarity(target_id, system.initial_user)
        initial_target_rank = system._target_rank(target_id, system.initial_user)
        final_target_rank = trajectory[-1]['target_rank'] if trajectory else initial_target_rank
        best_target_rank = min((step['target_rank'] for step in trajectory), default=initial_target_rank)
        results['initial_target_ranks'].append(initial_target_rank)
        results['final_target_ranks'].append(final_target_rank)
        results['best_target_ranks'].append(best_target_rank)

        if planner_mode != 'baseline' and hasattr(system, 'item_graph'):
            results['path_stats']['total_paths'] += 1
            if any(step.get('path_step') for step in trajectory):
                results['path_stats']['path_success_rate'] += hit
        if planner_mode != 'baseline':
            path_items = [step['rec'] for step in trajectory[:max(1, min(len(trajectory), config['MAX_PATH_LENGTH']))] if step['rec'] in news_embeds]
            path_pairs = list(zip(path_items[:-1], path_items[1:]))
            if path_pairs:
                pair_sims = [float(np.dot(news_embeds[a], news_embeds[b])) for a, b in path_pairs]
                target_bridge = float(np.dot(news_embeds[path_items[-1]], news_embeds[target_id])) if path_items else 0.0
                results['path_rationality'].append(float((sum(pair_sims) + target_bridge) / (len(pair_sims) + 1)))

        ratings_logs = np.full(20, initial_target_rating, dtype=np.float32)
        ranking_logs = np.full(20, initial_target_rank, dtype=np.float32)
        click_logs = np.zeros(20, dtype=np.float32)
        for round_idx, step in enumerate(trajectory[:20]):
            ratings_logs[round_idx] = step['target_rating']
            ranking_logs[round_idx] = step['target_rank']
            click_logs[round_idx] = 1.0 if step['accept'] else 0.0
        if len(trajectory) > 0:
            ratings_logs[len(trajectory):] = ratings_logs[len(trajectory) - 1]
            ranking_logs[len(trajectory):] = ranking_logs[len(trajectory) - 1]
        ratings_logs_diff = ratings_logs.copy()
        ratings_logs_diff[1:] = ratings_logs_diff[1:] - ratings_logs_diff[0]
        results['ratings_logs'].append(ratings_logs_diff)
        results['ranking_logs'].append(ranking_logs)
        results['click_logs'].append(click_logs)
        results['success_rounds'].append(success_round if success_round is not None else len(trajectory) + 1)
        accepts = sum(1 for step in trajectory if step['accept'])
        results['feedback_stats']['accept'] += accepts
        results['feedback_stats']['reject'] += len(trajectory) - accepts
        results['avg_rounds'] += num_rounds
        results['round_distribution'][num_rounds] += 1
        results['alpha_distribution'].append(system.alpha)
        results['trajectories'].append({'user_id': user_id, 'num_rounds': num_rounds, 'hit': hit})
        results['per_user_records'].append({
            'user_id': user_id,
            'history_signature': evaluated['history_signature'],
            'impression_signature': evaluated['impression_signature'],
            'target': target_id,
            'candidate_size': evaluated['candidate_size'],
            'initial_target_rank': initial_target_rank,
            'final_target_rank': final_target_rank,
            'best_target_rank': best_target_rank,
            'hit': hit,
            'rounds': num_rounds,
            'accepts': accepts,
            'acceptance_rate': float(accepts / max(len(trajectory), 1) * 100.0),
        })

    if valid_users > 0:
        results['avg_rounds'] = results['avg_rounds'] / valid_users
        total_feedback = results['feedback_stats']['accept'] + results['feedback_stats']['reject']
        results['acceptance_rate'] = results['feedback_stats']['accept'] / total_feedback * 100 if total_feedback > 0 else 0
        initial_ranks = np.array(results['initial_target_ranks'], dtype=np.int32) if results['initial_target_ranks'] else np.array([], dtype=np.int32)
        final_ranks = np.array(results['final_target_ranks'], dtype=np.int32) if results['final_target_ranks'] else np.array([], dtype=np.int32)
        best_ranks = np.array(results['best_target_ranks'], dtype=np.int32) if results['best_target_ranks'] else np.array([], dtype=np.int32)
        results['initial_target_rank_stats'] = {
            'mean': float(initial_ranks.mean()) if len(initial_ranks) > 0 else 0.0,
            'median': float(np.median(initial_ranks)) if len(initial_ranks) > 0 else 0.0,
            'min': int(initial_ranks.min()) if len(initial_ranks) > 0 else 0,
            'max': int(initial_ranks.max()) if len(initial_ranks) > 0 else 0,
            'rank_1_rate': float((initial_ranks == 1).mean() * 100) if len(initial_ranks) > 0 else 0.0,
            'top_5_rate': float((initial_ranks <= 5).mean() * 100) if len(initial_ranks) > 0 else 0.0,
            'top_10_rate': float((initial_ranks <= 10).mean() * 100) if len(initial_ranks) > 0 else 0.0,
        }
        results['final_target_rank_stats'] = {
            'mean': float(final_ranks.mean()) if len(final_ranks) > 0 else 0.0,
            'rank_1_rate': float((final_ranks == 1).mean() * 100) if len(final_ranks) > 0 else 0.0,
            'top_5_rate': float((final_ranks <= 5).mean() * 100) if len(final_ranks) > 0 else 0.0,
            'top_10_rate': float((final_ranks <= 10).mean() * 100) if len(final_ranks) > 0 else 0.0,
        }
        results['best_target_rank_stats'] = {
            'mean': float(best_ranks.mean()) if len(best_ranks) > 0 else 0.0,
            'rank_1_rate': float((best_ranks == 1).mean() * 100) if len(best_ranks) > 0 else 0.0,
            'top_5_rate': float((best_ranks <= 5).mean() * 100) if len(best_ranks) > 0 else 0.0,
            'top_10_rate': float((best_ranks <= 10).mean() * 100) if len(best_ranks) > 0 else 0.0,
        }
        if results['path_stats']['total_paths'] > 0:
            results['path_stats']['path_success_rate'] = results['path_stats']['path_success_rate'] / results['path_stats']['total_paths'] * 100
            if planner_mode == 'llm_path' and results['llm_stats']['success'] > 0:
                results['path_stats']['avg_path_length'] = results['llm_stats']['avg_path_length'] / results['llm_stats']['success']
            else:
                results['path_stats']['avg_path_length'] = float(np.mean([min(traj['num_rounds'], config['MAX_PATH_LENGTH']) for traj in results['trajectories']])) if results['trajectories'] else 0.0
        if planner_mode == 'llm_path':
            llm_called = results['llm_stats']['called']
            results['llm_stats']['success_rate'] = float(results['llm_stats']['success'] / llm_called * 100) if llm_called > 0 else 0.0
            results['llm_stats']['api_success_rate'] = float(results['llm_stats']['api_success'] / llm_called * 100) if llm_called > 0 else 0.0
            results['llm_stats']['remote_adopted_rate'] = float(results['llm_stats']['remote_adopted'] / llm_called * 100) if llm_called > 0 else 0.0
            results['llm_stats']['guard_adopted_rate'] = float(results['llm_stats']['guard_adopted'] / llm_called * 100) if llm_called > 0 else 0.0
            results['llm_stats']['rescue_adopted_rate'] = float(results['llm_stats']['rescue_adopted'] / llm_called * 100) if llm_called > 0 else 0.0
            results['llm_stats']['avg_path_length'] = results['path_stats']['avg_path_length']
        results['avg_path_rationality'] = float(np.mean(results['path_rationality'])) if results['path_rationality'] else 0.0
        ratings_logs_matrix = np.vstack(results['ratings_logs']) if results['ratings_logs'] else np.zeros((0, 20), dtype=np.float32)
        ranking_logs_matrix = np.vstack(results['ranking_logs']) if results['ranking_logs'] else np.zeros((0, 20), dtype=np.float32)
        click_logs_matrix = np.vstack(results['click_logs']) if results['click_logs'] else np.zeros((0, 20), dtype=np.float32)
        success_rounds = np.array(results['success_rounds'], dtype=np.int32) if results['success_rounds'] else np.array([], dtype=np.int32)
        eval_points = [p for p in config['K_LIST'] if p <= 20]
        for k in eval_points:
            idx = k - 1
            hit_at_k = float((success_rounds <= k).mean() * 100) if len(success_rounds) > 0 else 0.0
            ioi_at_k = float(ratings_logs_matrix[:, idx].mean()) if len(ratings_logs_matrix) > 0 else 0.0
            ior_at_k = float((ranking_logs_matrix[:, 0] - ranking_logs_matrix[:, idx]).mean()) if len(ranking_logs_matrix) > 0 else 0.0
            results['hr_at_k'][k] = hit_at_k
            results['ioi_at_k'][k] = ioi_at_k
            results['ior_at_k'][k] = ior_at_k
        rounds_array = np.array([record['rounds'] for record in results['per_user_records']], dtype=np.float32) if results['per_user_records'] else np.array([], dtype=np.float32)
        acceptance_array = np.array([record['acceptance_rate'] for record in results['per_user_records']], dtype=np.float32) if results['per_user_records'] else np.array([], dtype=np.float32)
        results['variance_stats'] = {
            'round_std': float(rounds_array.std()) if len(rounds_array) > 0 else 0.0,
            'accept_std': float(acceptance_array.std()) if len(acceptance_array) > 0 else 0.0,
            'hr_std_at_k': {},
            'ioi_std_at_k': {},
            'ior_std_at_k': {},
        }
        for k in eval_points:
            idx = k - 1
            hr_mask = (success_rounds <= k).astype(np.float32) if len(success_rounds) > 0 else np.array([], dtype=np.float32)
            rank_gain = (ranking_logs_matrix[:, 0] - ranking_logs_matrix[:, idx]) if len(ranking_logs_matrix) > 0 else np.array([], dtype=np.float32)
            results['variance_stats']['hr_std_at_k'][k] = float(hr_mask.std() * 100.0) if len(hr_mask) > 0 else 0.0
            results['variance_stats']['ioi_std_at_k'][k] = float(ratings_logs_matrix[:, idx].std()) if len(ratings_logs_matrix) > 0 else 0.0
            results['variance_stats']['ior_std_at_k'][k] = float(rank_gain.std()) if len(rank_gain) > 0 else 0.0
        results['hit_rate'] = results['hr_at_k'].get(max(eval_points), 0.0) if eval_points else 0.0
        results['valid_users'] = valid_users
    return results


def metric_is_monotonic(metric_dict: dict, increasing: bool = True) -> bool:
    ordered_values = [metric_dict[k] for k in sorted(metric_dict.keys())]
    if len(ordered_values) <= 1:
        return True
    if increasing:
        return all(left <= right + 1e-8 for left, right in zip(ordered_values[:-1], ordered_values[1:]))
    return all(left >= right - 1e-8 for left, right in zip(ordered_values[:-1], ordered_values[1:]))


def _method_record_map(results: dict, field: str) -> dict:
    return {record['user_id']: record[field] for record in results.get('per_user_records', [])}


def _metric_at_k(results: dict, metric_name: str, k: int) -> float:
    return float(results.get(metric_name, {}).get(k, 0.0))


def _primary_k(config: dict) -> int:
    return int(max(config.get('K_LIST', [config.get('MAX_ROUNDS', 6)])))


def _secondary_k(config: dict) -> int:
    k_list = sorted(int(k) for k in config.get('K_LIST', []))
    if len(k_list) >= 2:
        return k_list[-2]
    return _primary_k(config)


def _is_finite_triplet(results: dict) -> bool:
    primary_k = max(results.get('hr_at_k', {}).keys(), default=0)
    values = [
        _metric_at_k(results, 'hr_at_k', primary_k),
        _metric_at_k(results, 'ioi_at_k', primary_k),
        _metric_at_k(results, 'ior_at_k', primary_k),
    ]
    return all(np.isfinite(value) for value in values)


def _has_nontrivial_signal(results: dict) -> bool:
    primary_k = max(results.get('hr_at_k', {}).keys(), default=0)
    return (
        _metric_at_k(results, 'hr_at_k', primary_k) > 0.0
        or abs(_metric_at_k(results, 'ioi_at_k', primary_k)) > 1e-8
        or abs(_metric_at_k(results, 'ior_at_k', primary_k)) > 1e-8
    )


def build_overall_table(results_by_method: dict, config: dict) -> List[dict]:
    primary_k = _primary_k(config)
    secondary_k = _secondary_k(config)
    return [
        {
            'Method': method_name,
            f'HR@{secondary_k}': round(_metric_at_k(results, 'hr_at_k', secondary_k), 4),
            f'HR@{primary_k}': round(_metric_at_k(results, 'hr_at_k', primary_k), 4),
            f'IOI@{primary_k}': round(_metric_at_k(results, 'ioi_at_k', primary_k), 6),
            f'IOR@{primary_k}': round(_metric_at_k(results, 'ior_at_k', primary_k), 4),
            'Accept %': round(float(results.get('acceptance_rate', 0.0)), 4),
            'Avg Round': round(float(results.get('avg_rounds', 0.0)), 4),
            'Round Std': round(float(results.get('variance_stats', {}).get('round_std', 0.0)), 4),
            'Accept Std': round(float(results.get('variance_stats', {}).get('accept_std', 0.0)), 4),
        }
        for method_name, results in results_by_method.items()
    ]


def build_ranking_table(results_by_method: dict) -> List[dict]:
    rows = []
    for method_name, results in results_by_method.items():
        init_rank_stats = results.get('initial_target_rank_stats', {})
        final_rank_stats = results.get('final_target_rank_stats', {})
        best_rank_stats = results.get('best_target_rank_stats', {})
        rows.append({
            'Method': method_name,
            'Init Mean Rank': round(float(init_rank_stats.get('mean', 0.0)), 4),
            'Final Mean Rank': round(float(final_rank_stats.get('mean', 0.0)), 4),
            'Best Mean Rank': round(float(best_rank_stats.get('mean', 0.0)), 4),
            'Final Top-10 %': round(float(final_rank_stats.get('top_10_rate', 0.0)), 4),
        })
    return rows


def build_path_behavior_table(results_by_method: dict) -> List[dict]:
    rows = []
    preferred_order = [
        BASELINE_METHOD_NAME,
        IDEA_METHOD_NAME,
        LLM_METHOD_NAME,
        LLM_NO_GUARD_METHOD_NAME,
        LLM_NO_REFLECTION_METHOD_NAME,
        LLM_NO_RESCUE_METHOD_NAME,
    ]
    ordered_method_names = [name for name in preferred_order if name in results_by_method]
    ordered_method_names.extend([name for name in results_by_method.keys() if name not in ordered_method_names])
    for method_name in ordered_method_names:
        if method_name not in results_by_method:
            continue
        results = results_by_method[method_name]
        llm_stats = results.get('llm_stats', {})
        remote_adopted = 0.0
        guard_adopted = 0.0
        rescue_adopted = 0.0
        if method_name == BASELINE_METHOD_NAME:
            valid_path = 0.0
            api_success = 0.0
            fallback = 0.0
            stability = 'no_path_planner'
        elif method_name == IDEA_METHOD_NAME:
            valid_path = 100.0
            api_success = 0.0
            fallback = 0.0
            stability = 'deterministic_path'
        else:
            valid_path = round(float(llm_stats.get('success_rate', 0.0)), 4)
            api_success = round(float(llm_stats.get('api_success_rate', 0.0)), 4)
            remote_adopted = round(float(llm_stats.get('remote_adopted_rate', 0.0)), 4)
            guard_adopted = round(float(llm_stats.get('guard_adopted_rate', 0.0)), 4)
            rescue_adopted = round(float(llm_stats.get('rescue_adopted_rate', 0.0)), 4)
            called = float(llm_stats.get('called', 0.0))
            fallback = round(float(llm_stats.get('fallback', 0.0)) / called * 100.0, 4) if called > 0 else 0.0
            if api_success == 0.0 and valid_path > 0.0:
                stability = 'rescue_only'
            elif fallback == 0.0 and valid_path >= 99.99 and api_success >= 99.99:
                stability = 'high'
            else:
                stability = 'check'
        rows.append({
            'Method': method_name,
            'Avg Path Length': round(float(results.get('path_stats', {}).get('avg_path_length', 0.0)), 4),
            'Path Selected %': valid_path,
            'API Success %': api_success,
            'Remote Adopt %': remote_adopted if method_name == LLM_METHOD_NAME else 0.0,
            'Guard Adopt %': guard_adopted if method_name == LLM_METHOD_NAME else 0.0,
            'Rescue Adopt %': rescue_adopted if method_name == LLM_METHOD_NAME else 0.0,
            'Fallback %': fallback,
            'Stability': stability,
        })
    return rows


def build_exp1_checks(results_by_method: dict, eval_sessions: List[dict], config: dict) -> dict:
    expected_user_ids = {session['user_id'] for session in eval_sessions}
    expected_history_signatures = {session['user_id']: session['history_signature'] for session in eval_sessions}
    expected_impression_signatures = {session['user_id']: session['impression_signature'] for session in eval_sessions}

    method_user_ids = {}
    method_history_signatures = {}
    method_candidate_signatures = {}
    method_targets = {}
    for method_name, results in results_by_method.items():
        records = results.get('per_user_records', [])
        method_user_ids[method_name] = {record['user_id'] for record in records}
        method_history_signatures[method_name] = _method_record_map(results, 'history_signature')
        method_candidate_signatures[method_name] = _method_record_map(results, 'impression_signature')
        method_targets[method_name] = _method_record_map(results, 'target')

    same_input_sessions = all(user_ids == expected_user_ids for user_ids in method_user_ids.values())
    same_history_inputs = all(history_map == expected_history_signatures for history_map in method_history_signatures.values())
    same_candidate_sets = all(candidate_map == expected_impression_signatures for candidate_map in method_candidate_signatures.values())

    baseline_targets = method_targets.get(BASELINE_METHOD_NAME, {})
    same_targets = all(target_map == baseline_targets for method_name, target_map in method_targets.items() if method_name != BASELINE_METHOD_NAME)

    backbone_results = results_by_method.get(BASELINE_METHOD_NAME, {})
    greedy_results = results_by_method.get(IDEA_METHOD_NAME, {})
    llm_results = results_by_method.get(LLM_METHOD_NAME, {})
    llm_stats = llm_results.get('llm_stats', {})

    hr_monotonic = {
        'baseline': metric_is_monotonic(backbone_results.get('hr_at_k', {}), increasing=True),
        'greedy': metric_is_monotonic(greedy_results.get('hr_at_k', {}), increasing=True),
        'llm': metric_is_monotonic(llm_results.get('hr_at_k', {}), increasing=True),
    }
    ioi_monotonic = {
        'baseline': metric_is_monotonic(backbone_results.get('ioi_at_k', {}), increasing=True),
        'greedy': metric_is_monotonic(greedy_results.get('ioi_at_k', {}), increasing=True),
        'llm': metric_is_monotonic(llm_results.get('ioi_at_k', {}), increasing=True),
    }
    ior_monotonic = {
        'baseline': metric_is_monotonic(backbone_results.get('ior_at_k', {}), increasing=True),
        'greedy': metric_is_monotonic(greedy_results.get('ior_at_k', {}), increasing=True),
        'llm': metric_is_monotonic(llm_results.get('ior_at_k', {}), increasing=True),
    }

    primary_k = _primary_k(config)
    greedy_vs_backbone_hr = _metric_at_k(greedy_results, 'hr_at_k', primary_k) - _metric_at_k(backbone_results, 'hr_at_k', primary_k)
    llm_vs_backbone_hr = _metric_at_k(llm_results, 'hr_at_k', primary_k) - _metric_at_k(backbone_results, 'hr_at_k', primary_k)
    greedy_vs_llm_hr = _metric_at_k(greedy_results, 'hr_at_k', primary_k) - _metric_at_k(llm_results, 'hr_at_k', primary_k)

    return {
        'primary_eval_k': primary_k,
        'session_count': len(eval_sessions),
        'same_input_sessions': same_input_sessions,
        'same_history_inputs': same_history_inputs,
        'same_candidate_sets': same_candidate_sets,
        'same_targets': same_targets,
        'llm_valid_path_100': float(llm_stats.get('success_rate', 0.0)) >= 99.99,
        'llm_api_success_nonzero': float(llm_stats.get('api_success_rate', 0.0)) > 0.0,
        'llm_fallback_zero': int(llm_stats.get('fallback', 0)) == 0,
        'llm_source_accounted': abs(
            float(llm_stats.get('remote_adopted_rate', 0.0))
            + float(llm_stats.get('guard_adopted_rate', 0.0))
            + float(llm_stats.get('rescue_adopted_rate', 0.0))
            - float(llm_stats.get('success_rate', 0.0))
        ) <= 1e-6,
        'hr_monotonic': hr_monotonic,
        'ioi_monotonic': ioi_monotonic,
        'ior_monotonic': ior_monotonic,
        'llm_metrics_finite': _is_finite_triplet(llm_results),
        'llm_not_all_zero': _has_nontrivial_signal(llm_results),
        'greedy_metrics_finite': _is_finite_triplet(greedy_results),
        'backbone_metrics_finite': _is_finite_triplet(backbone_results),
        f'hr@{primary_k}_deltas': {
            'greedy_minus_backbone': round(greedy_vs_backbone_hr, 4),
            'llm_minus_backbone': round(llm_vs_backbone_hr, 4),
            'greedy_minus_llm': round(greedy_vs_llm_hr, 4),
        },
        'greedy_not_abnormally_dominant': greedy_vs_backbone_hr <= 20.0 and greedy_vs_llm_hr <= 20.0,
        'llm_not_completely_random': _has_nontrivial_signal(llm_results) and _is_finite_triplet(llm_results),
    }


def print_exp1_tables(overall_table: List[dict], ranking_table: List[dict], path_table: List[dict], checks: dict, config: dict):
    primary_k = _primary_k(config)
    secondary_k = _secondary_k(config)
    print('\n' + '=' * 80)
    print('EXP1 Table 1: Smoke Monitoring Metrics')
    print('=' * 80)
    print(f"{'Method':<24} {f'HR@{secondary_k}':<10} {f'HR@{primary_k}':<10} {f'IOI@{primary_k}':<12} {f'IOR@{primary_k}':<12} {'Accept%':<10} {'AvgRnd':<10} {'RndStd':<10}")
    for row in overall_table:
        print(f"{row['Method']:<24} {row[f'HR@{secondary_k}']:<10.4f} {row[f'HR@{primary_k}']:<10.4f} {row[f'IOI@{primary_k}']:<12.6f} {row[f'IOR@{primary_k}']:<12.4f} {row['Accept %']:<10.4f} {row['Avg Round']:<10.4f} {row['Round Std']:<10.4f}")
    print('\n' + '=' * 80)
    print('EXP1 Table 2: Target Rank Evolution')
    print('=' * 80)
    print(f"{'Method':<24} {'InitMean':<12} {'FinalMean':<12} {'BestMean':<12} {'FinalTop10%':<12}")
    for row in ranking_table:
        print(f"{row['Method']:<24} {row['Init Mean Rank']:<12.4f} {row['Final Mean Rank']:<12.4f} {row['Best Mean Rank']:<12.4f} {row['Final Top-10 %']:<12.4f}")
    print('\n' + '=' * 80)
    print('EXP1 Table 3: Path Behavior Analysis')
    print('=' * 80)
    print(f"{'Method':<24} {'AvgLen':<8} {'PathSel%':<10} {'APISucc%':<10} {'Remote%':<10} {'Guard%':<10} {'Rescue%':<10} {'Fallback%':<10} {'Stability':<12}")
    for row in path_table:
        print(f"{row['Method']:<24} {row['Avg Path Length']:<8.4f} {row['Path Selected %']:<10.4f} {row['API Success %']:<10.4f} {row['Remote Adopt %']:<10.4f} {row['Guard Adopt %']:<10.4f} {row['Rescue Adopt %']:<10.4f} {row['Fallback %']:<10.4f} {row['Stability']:<12}")
    print('\nEXP1 checks:')
    for key, value in checks.items():
        print(f'  {key}: {value}')


def run_exp1_pipeline(script_dir: str, data_loader, news_embeds, embed_type: str, config: dict):
    if config['ENABLE_STEP3'] and config['RUN_LLM_PATH']:
        llm_config_status = validate_llm_configuration()
        print(f"LLM preflight: backend={llm_config_status['backend']} model={llm_config_status['model']}")
        if not llm_config_status['ok']:
            raise RuntimeError(
                'LLM preflight failed before EXP1 execution. '
                f"backend={llm_config_status['backend']} base_url={llm_config_status['base_url']} error={llm_config_status['error']}"
            )

    eval_sessions = build_eval_sessions(config['SAMPLE_SIZE'], data_loader, news_embeds, config['IMPRESSION_SIZE'], config['RANDOM_SEED'])
    print(f'EXP1 固定评测 sessions: {len(eval_sessions)} 用户')

    backbone_results = {}
    greedy_results = {}
    llm_results = None
    llm_ablation_results = {}

    print(f'开始 EXP1 对照实验 ({len(eval_sessions)} 用户)')
    if config['RUN_BASELINE']:
        backbone_results = run_experiment('baseline', BASELINE_METHOD_NAME, eval_sessions, config, news_embeds, data_loader)
    if config['RUN_GREEDY_PATH']:
        greedy_results = run_experiment('greedy_path', IDEA_METHOD_NAME, eval_sessions, config, news_embeds, data_loader)
    if config['ENABLE_STEP3'] and config['RUN_LLM_PATH']:
        llm_results = run_experiment('llm_path', LLM_METHOD_NAME, eval_sessions, config, news_embeds, data_loader)
    if config['ENABLE_STEP3'] and config.get('RUN_INTERNAL_ABLATIONS', False):
        if config.get('RUN_LLM_NO_GUARD', False):
            llm_ablation_results[LLM_NO_GUARD_METHOD_NAME] = run_experiment(
                'llm_path',
                LLM_NO_GUARD_METHOD_NAME,
                eval_sessions,
                config,
                news_embeds,
                data_loader,
                overrides={'enable_llm_guard': False},
            )
        if config.get('RUN_LLM_NO_REFLECTION', False):
            llm_ablation_results[LLM_NO_REFLECTION_METHOD_NAME] = run_experiment(
                'llm_path',
                LLM_NO_REFLECTION_METHOD_NAME,
                eval_sessions,
                config,
                news_embeds,
                data_loader,
                overrides={'enable_reflection': False},
            )
        if config.get('RUN_LLM_NO_RESCUE', False):
            llm_ablation_results[LLM_NO_RESCUE_METHOD_NAME] = run_experiment(
                'llm_path',
                LLM_NO_RESCUE_METHOD_NAME,
                eval_sessions,
                config,
                news_embeds,
                data_loader,
                overrides={'enable_llm_rescue': False},
            )

    results_by_method = {}
    if config['RUN_BASELINE']:
        results_by_method[BASELINE_METHOD_NAME] = backbone_results
    if config['RUN_GREEDY_PATH']:
        results_by_method[IDEA_METHOD_NAME] = greedy_results
    if llm_results is not None:
        results_by_method[LLM_METHOD_NAME] = llm_results
    for method_name, ablation_result in llm_ablation_results.items():
        results_by_method[method_name] = ablation_result

    overall_table = build_overall_table(results_by_method, config)
    ranking_table = build_ranking_table(results_by_method)
    path_behavior_table = build_path_behavior_table(results_by_method)
    exp1_checks = build_exp1_checks(results_by_method, eval_sessions, config)
    print_exp1_tables(overall_table, ranking_table, path_behavior_table, exp1_checks, config)

    output_file = os.path.join(script_dir, 'results_exp1.json')
    exp1_output = {
        'experiment_name': config['EXP1_NAME'],
        'claim': 'We first conduct a sanity-check experiment to verify that all compared methods operate under consistent conditions and produce stable outputs.',
        'experiment_config': {
            'sample_size': config['SAMPLE_SIZE'],
            'random_seed': config['RANDOM_SEED'],
            'round': config['MAX_ROUNDS'],
            'candidate_k': config['LLM_CANDIDATE_K'],
            'graph_k': config['GRAPH_K'],
            'prompt_version': config['EXP1_PROMPT_VERSION'],
            'impression_size': config['IMPRESSION_SIZE'],
            'max_path_length': config['MAX_PATH_LENGTH'],
            'llm_hybrid_rounds': config['LLM_HYBRID_ROUNDS'],
            'llm_path_k': config['LLM_PATH_K'],
            'llm_target_prior': config['LLM_TARGET_PRIOR'],
            'llm_safe_margin': config['LLM_SAFE_MARGIN'],
            'llm_release_rank': config['LLM_RELEASE_RANK'],
            'lambda0': config['LAMBDA0'],
            'lambda1': config['LAMBDA1'],
            'enable_gating': config['ENABLE_GATING'],
            'enable_replanning': config['ENABLE_REPLANNING'],
            'enable_reflection': config['ENABLE_REFLECTION'],
            'run_baseline': config['RUN_BASELINE'],
            'run_greedy_path': config['RUN_GREEDY_PATH'],
            'run_llm_path': config['RUN_LLM_PATH'],
        },
        'embedding_info': {'type': embed_type, 'total_items': len(news_embeds)},
        'fairness_checks': exp1_checks,
        'tables': {
            'table_1_overall_performance': overall_table,
            'table_2_ranking_state_quality': ranking_table,
            'table_3_path_behavior_analysis': path_behavior_table,
        },
        'backbone_results': backbone_results if config['RUN_BASELINE'] else None,
        'greedy_results': greedy_results if config['RUN_GREEDY_PATH'] else None,
        'llm_results': llm_results,
        'llm_ablation_results': llm_ablation_results,
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(exp1_output), f, indent=2, ensure_ascii=False)
    print(f'\nEXP1 结果保存到: {output_file}')
    return exp1_output


def parse_args():
    parser = argparse.ArgumentParser(description='Run EXP1 target-aware multi-round recommendation experiment.')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of users to evaluate.')
    parser.add_argument('--max_rounds', type=int, default=None, help='Maximum interaction rounds per user.')
    parser.add_argument('--data_dir', type=str, default=None, help='Optional path to MIND data directory.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed override.')
    parser.add_argument('--disable_parallel', action='store_true', help='Disable parallel session execution.')
    parser.add_argument('--run_internal_ablations', action='store_true', help='Run Ours internal ablations.')
    parser.add_argument('--run_llm_no_guard', action='store_true', help='Run ablation without LLM guard.')
    parser.add_argument('--run_llm_no_reflection', action='store_true', help='Run ablation without reflection.')
    parser.add_argument('--run_llm_no_rescue', action='store_true', help='Run ablation without rescue.')
    return parser.parse_args()


def _build_cli_config(args):
    config = dict(DEFAULT_EXP1_CONFIG)
    if args.sample_size is not None:
        config['SAMPLE_SIZE'] = args.sample_size
    if args.max_rounds is not None:
        config['MAX_ROUNDS'] = args.max_rounds
    if args.seed is not None:
        config['RANDOM_SEED'] = args.seed
    if args.disable_parallel:
        config['EXP1_PARALLEL_EXECUTION'] = False
    if args.run_internal_ablations:
        config['RUN_INTERNAL_ABLATIONS'] = True
        config['RUN_LLM_NO_GUARD'] = True
        config['RUN_LLM_NO_REFLECTION'] = True
        config['RUN_LLM_NO_RESCUE'] = True
    if args.run_llm_no_guard:
        config['RUN_INTERNAL_ABLATIONS'] = True
        config['RUN_LLM_NO_GUARD'] = True
    if args.run_llm_no_reflection:
        config['RUN_INTERNAL_ABLATIONS'] = True
        config['RUN_LLM_NO_REFLECTION'] = True
    if args.run_llm_no_rescue:
        config['RUN_INTERNAL_ABLATIONS'] = True
        config['RUN_LLM_NO_RESCUE'] = True
    return config


def main():
    from data.data_loader import MINDDataLoader
    from data.embeddings import build_news_embeddings

    args = parse_args()
    configure_stdout_utf8()
    print('EXP1 CLI 启动...')
    set_global_seed(GLOBAL_SEED)
    install_required_packages()
    print_gpu_info()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or resolve_data_dir(script_dir)
    validate_data_files(data_dir)

    print(f'使用数据目录: {data_dir}')
    data_loader = MINDDataLoader(data_dir, min_interactions=5, max_history=30)
    all_items = sorted(data_loader.news.keys())
    news_embeds, embed_type = build_news_embeddings(data_dir, data_loader, all_items, use_minilm=True)

    config = _build_cli_config(args)
    set_global_seed(config['RANDOM_SEED'])
    print(f"配置: sample_size={config['SAMPLE_SIZE']} max_rounds={config['MAX_ROUNDS']} seed={config['RANDOM_SEED']}")
    run_exp1_pipeline(script_dir, data_loader, news_embeds, embed_type, config)


if __name__ == '__main__':
    main()
