import json
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import get_planner_engine
from planner_prompt import build_reflection_rules, build_target_oriented_planner_prompt
from runtime_utils import stable_seed_from_text


class ONeRecBaseline:
    def __init__(
        self,
        news_embeds: Dict[str, np.ndarray],
        categories: Dict[str, str],
        item_titles: Optional[Dict[str, str]],
        user_history: List[str],
        impressions: List[str],
        reach_min_sim: float = 0.05,
        feedback_threshold: float = 0.5,
        random_seed: Optional[int] = None,
        debug: bool = False,
        planner_mode: str = 'greedy_path',
        graph_k: int = 10,
        max_path_length: int = 5,
        llm_candidate_k: int = 20,
        llm_path_k: int = 5,
        llm_target_prior: float = 0.65,
        llm_hybrid_rounds: int = 2,
        llm_safe_margin: float = 0.015,
        llm_release_rank: int = 3,
        enable_llm_guard: bool = True,
        enable_llm_rescue: bool = True,
        enable_gating: bool = True,
        enable_replanning: bool = True,
        enable_reflection: bool = True,
    ):
        self.news = news_embeds
        self.cat = categories
        self.history = user_history.copy()
        self.item_titles = item_titles or {}
        self.threshold = feedback_threshold
        self.reach_min = reach_min_sim
        self.debug = debug
        self.planner_mode = planner_mode
        self.use_path_planner = planner_mode in {'greedy_path', 'llm_path'}
        self.use_llm_planner = planner_mode == 'llm_path'
        self.graph_k = graph_k
        self.max_path_length = max_path_length
        self.llm_candidate_k = llm_candidate_k
        self.llm_path_k = llm_path_k
        self.llm_target_prior = llm_target_prior
        self.llm_hybrid_rounds = llm_hybrid_rounds
        self.llm_safe_margin = llm_safe_margin
        self.llm_release_rank = llm_release_rank
        self.enable_llm_guard = enable_llm_guard
        self.enable_llm_rescue = enable_llm_rescue
        self.enable_gating = enable_gating
        self.enable_replanning = enable_replanning
        self.enable_reflection = enable_reflection
        session_key = '|'.join(user_history) + '||' + '|'.join(impressions)
        self.random_seed = random_seed if random_seed is not None else stable_seed_from_text(session_key)
        self.rng = np.random.default_rng(self.random_seed)
        self.planner_llm = None
        self.llm_feedback_cache = {}
        self.llm_feedback_calls = 0
        self.llm_feedback_cache_hits = 0
        self.llm_feedback_parse_failures = 0
        self.llm_called = 0
        self.llm_api_success = 0
        self.llm_api_failure = 0
        self.llm_api_error_messages = []
        self.llm_invalid_path = 0
        self.llm_success = 0
        self.llm_fallback = 0
        self.llm_remote_adopted = 0
        self.llm_guard_adopted = 0
        self.llm_rescue_adopted = 0
        self.llm_path_lengths = []
        self.llm_path_length_sum = 0
        self.failed_paths = []
        self.reject_streak = 0
        self.last_llm_reason = ''
        self.last_llm_stage = ''
        self.policy_bias = {
            'bridge_weight': 1.0,
            'target_weight': 1.0,
            'history_weight': 1.0,
        }
        self.temporal_decay = 0.35
        self.user_update_eta = 0.20
        self.temporal_state_mix = 0.55
        self.intent_graph_summary_k = 3
        self.current_target_id = None
        self.current_graph_nodes = []
        self.impressions = [i for i in impressions if i in news_embeds]
        base_graph_nodes = list(dict.fromkeys(self.history + self.impressions))
        self.item_graph = {}
        self.current_graph_nodes = base_graph_nodes
        if self.use_path_planner:
            self.item_graph = self._build_item_graph(base_graph_nodes)
            if self.debug:
                print(f'  path graph ready: {len(self.item_graph)} nodes, K={self.graph_k}')

        self.user = self._compute_temporal_user_state(self.history)

        self.lambda0 = 0.1
        self.lambda1 = 0.2
        self.eps = 1e-8
        self.target_percentile_low = 0.2
        self.target_percentile_high = 0.45
        self.feedback_scale = 2.0
        self.feedback_bias = 1.0
        self.reject_decay = 0.0
        self.alpha_min = 0.1
        self.alpha_max = 0.55
        self.alpha_w1 = 0.5
        self.alpha_w2 = 0.5
        self.alpha_fn = True
        self.initial_user = self.user.copy()
        self.alpha = self._compute_openness()
        self.path_min_progress = 0.01
        self.path_monotonic_penalty = 2.5
        self.path_length_penalty = 0.08
        self.path_first_step_weight = 0.30
        self.path_progress_weight = 0.70
        self.baseline_confidence_threshold = 0.035
        self.path_override_margin = 0.02

    def _item_text(self, item_id: str) -> str:
        title = self.item_titles.get(item_id, item_id)
        category = self.cat.get(item_id, '')
        return f'{item_id} | {title} | {category}' if category else f'{item_id} | {title}'

    def _valid_item_ids(self, item_ids: Optional[List[str]] = None) -> List[str]:
        source_ids = self.history if item_ids is None else item_ids
        return [item_id for item_id in source_ids if item_id in self.news]

    def _temporal_attention_weights(self, item_ids: List[str]) -> np.ndarray:
        valid_item_ids = self._valid_item_ids(item_ids)
        if not valid_item_ids:
            return np.array([], dtype=np.float32)
        history_length = len(valid_item_ids)
        positions = np.arange(history_length - 1, -1, -1, dtype=np.float32)
        weights = np.exp(-self.temporal_decay * positions)
        weights_sum = float(np.sum(weights))
        if weights_sum <= 0.0:
            return np.full(history_length, 1.0 / history_length, dtype=np.float32)
        return (weights / weights_sum).astype(np.float32)

    def _compute_temporal_user_state(self, history_ids: Optional[List[str]] = None) -> np.ndarray:
        valid_history = self._valid_item_ids(history_ids)
        if not valid_history:
            dim = next(iter(self.news.values())).shape[0]
            return np.zeros(dim, dtype=np.float32)
        weights = self._temporal_attention_weights(valid_history)
        history_matrix = np.stack([self.news[item_id] for item_id in valid_history], axis=0)
        weighted_state = np.sum(history_matrix * weights[:, None], axis=0)
        return self._normalize(weighted_state)

    def _build_user_profile(self) -> str:
        recent_items = self.history[-5:]
        recent_weights = self._temporal_attention_weights(recent_items)
        if recent_items and len(recent_weights) == len([item_id for item_id in recent_items if item_id in self.news]):
            valid_recent_items = [item_id for item_id in recent_items if item_id in self.news]
            recent_text = '; '.join(
                f'{self._item_text(item_id)} (w={weight:.2f})'
                for item_id, weight in zip(valid_recent_items, recent_weights)
            )
        else:
            recent_text = '; '.join(self._item_text(item_id) for item_id in recent_items) if recent_items else 'N/A'
        category_counts = defaultdict(int)
        for item_id in self.history:
            category_counts[self.cat.get(item_id, 'unknown')] += 1
        top_categories = sorted(category_counts.items(), key=lambda pair: (-pair[1], pair[0]))[:3]
        category_text = ', '.join(f'{cat}:{count}' for cat, count in top_categories) if top_categories else 'N/A'
        return f'recent_history=[{recent_text}] | top_categories=[{category_text}]'

    def _sigmoid(self, value):
        return 1.0 / (1.0 + np.exp(-np.clip(value, -10, 10)))

    def _normalize(self, vector):
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _compute_openness(self) -> float:
        if len(self.history) < 2:
            return self.lambda0
        transitions = sum(
            1 for i in range(len(self.history) - 1)
            if self.cat.get(self.history[i], '') != self.cat.get(self.history[i + 1], '')
        )
        frequency = transitions / (len(self.history) - 1)
        vecs = [self.news[i] for i in self.history if i in self.news]
        if len(vecs) >= 2:
            sims = cosine_similarity(vecs)
            upper = sims[np.triu_indices(len(vecs), 1)]
            diversity = float(1.0 - np.mean(upper))
        else:
            diversity = 0.0
        return float(np.clip(self.lambda0 + self.lambda1 * (frequency + diversity), self.alpha_min, self.alpha_max))

    def _compute_openness_fine_grained(self, item_id: str) -> float:
        item_vec = self.news[item_id]
        hist_vecs = [self.news[i] for i in self.history if i in self.news]
        user_centroid = self._compute_temporal_user_state(self.history) if hist_vecs else self.user
        if len(hist_vecs) >= 2:
            shifts = []
            for idx in range(len(hist_vecs) - 1):
                shift = 1.0 - cosine_similarity([hist_vecs[idx]], [hist_vecs[idx + 1]])[0][0]
                shifts.append(shift)
            intent_shift = float(np.mean(shifts))
        else:
            intent_shift = 0.0
        semantic_distance = 1.0 - cosine_similarity([user_centroid], [item_vec])[0][0]
        alpha = self._sigmoid(self.alpha_w1 * intent_shift + self.alpha_w2 * float(semantic_distance))
        return float(np.clip(alpha, self.alpha_min, self.alpha_max))

    def _build_item_graph(self, node_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        graph_nodes = node_ids if node_ids is not None else self.impressions
        unique_nodes = [item_id for item_id in dict.fromkeys(graph_nodes) if item_id in self.news]
        graph = {}
        for item_id in unique_nodes:
            item_vec = self.news[item_id]
            similarities = []
            for other_id in unique_nodes:
                if other_id != item_id and other_id in self.news:
                    other_vec = self.news[other_id]
                    sim = float(cosine_similarity([item_vec], [other_vec])[0][0])
                    similarities.append((other_id, sim))
            similarities.sort(key=lambda x: -x[1])
            graph[item_id] = [neighbor_id for neighbor_id, _ in similarities[:self.graph_k]]
        return graph

    def _refresh_intention_graph(self, target_id: Optional[str] = None, candidate_items: Optional[List[str]] = None) -> Dict[str, List[str]]:
        candidate_pool = list(candidate_items) if candidate_items is not None else list(self.impressions)
        graph_nodes = list(dict.fromkeys(self.history + candidate_pool + ([target_id] if target_id and target_id in self.news else [])))
        self.current_target_id = target_id
        self.current_graph_nodes = graph_nodes
        self.item_graph = self._build_item_graph(graph_nodes)
        return self.item_graph

    def _build_graph_topology_summary(self, target_id: str, candidate_items: Optional[List[str]] = None) -> str:
        if not self.item_graph:
            self._refresh_intention_graph(target_id, candidate_items)
        anchor_nodes = list(dict.fromkeys(self.history[-2:] + (list(candidate_items)[:3] if candidate_items else []) + [target_id]))
        summary_lines = []
        for item_id in anchor_nodes:
            if item_id not in self.item_graph:
                continue
            neighbors = self.item_graph.get(item_id, [])[:self.intent_graph_summary_k]
            if neighbors:
                summary_lines.append(f'- {item_id} -> {", ".join(neighbors)}')
        return '\n'.join(summary_lines) if summary_lines else '- topology unavailable'

    def _path_structure_metrics(self, path: List[str], target_id: str, user_vec: Optional[np.ndarray] = None) -> Dict[str, float]:
        if not path or target_id not in self.news:
            return {
                'semantic_coherence': 0.0,
                'target_proximity': 0.0,
                'history_consistency': 0.0,
                'mean_progress': 0.0,
                'monotonic_violations': 0.0,
            }
        anchor_user = self.user if user_vec is None else user_vec
        path_pairs = list(zip(path[:-1], path[1:]))
        pair_sims = [self._cosine(self.news[left], self.news[right]) for left, right in path_pairs if left in self.news and right in self.news]
        semantic_coherence = float(np.mean(pair_sims)) if pair_sims else 1.0
        final_step_id = path[-2] if len(path) >= 2 else path[-1]
        target_proximity = self._cosine(self.news[final_step_id], self.news[target_id]) if final_step_id in self.news else 0.0
        target_sims = self._path_target_sims(path, target_id)
        progress_deltas = [target_sims[idx + 1] - target_sims[idx] for idx in range(len(target_sims) - 1)]
        history_consistency = float(np.mean([
            self._cosine(self.news[item_id], anchor_user)
            for item_id in path if item_id in self.news
        ])) if path else 0.0
        return {
            'semantic_coherence': semantic_coherence,
            'target_proximity': target_proximity,
            'history_consistency': history_consistency,
            'mean_progress': float(np.mean([max(delta, 0.0) for delta in progress_deltas])) if progress_deltas else 0.0,
            'monotonic_violations': float(sum(1 for delta in progress_deltas if delta <= self.path_min_progress)),
        }

    def _format_candidate_paths(self, candidate_paths: List[List[str]], target_id: str) -> str:
        if not candidate_paths:
            return '1. ' + target_id
        formatted_lines = []
        for idx, path in enumerate(candidate_paths):
            metrics = self._path_structure_metrics(path, target_id)
            formatted_lines.append(
                f'{idx + 1}. ' + ' -> '.join(path)
                + f' | coherence={metrics["semantic_coherence"]:.3f}'
                + f' | target_proximity={metrics["target_proximity"]:.3f}'
                + f' | history_consistency={metrics["history_consistency"]:.3f}'
            )
        return '\n'.join(formatted_lines)

    def _plan_path(self, user_vec: np.ndarray, target_id: str) -> List[str]:
        self._refresh_intention_graph(target_id, list(self.impressions))
        if target_id not in self.news or not hasattr(self, 'item_graph'):
            return [target_id]
        start_candidates = []
        for item_id in self.impressions:
            if item_id == target_id or item_id not in self.news:
                continue
            start_candidates.append((item_id, self._score(item_id, target_id)))
        if not start_candidates:
            return [target_id]
        start_candidates.sort(key=lambda pair: -pair[1])
        start_id = start_candidates[0][0]
        return self._plan_path_from_start(start_id, target_id, list(self.impressions))

    def _rank_baseline_candidates(self, target_id: str) -> List[tuple[str, float]]:
        ranked = []
        for item_id in self.impressions:
            ranked.append((item_id, self._score(item_id, target_id)))
        ranked.sort(key=lambda pair: -pair[1])
        return ranked

    def _select_baseline_first_rec(self, round_idx: int, target_id: str, planned_path: Optional[List[str]], path_cursor: int) -> str:
        ranked_items = self._rank_baseline_candidates(target_id)
        if not ranked_items:
            return target_id
        baseline_rec = ranked_items[1][0] if round_idx == 0 and len(ranked_items) > 1 else ranked_items[0][0]
        baseline_score = ranked_items[1][1] if round_idx == 0 and len(ranked_items) > 1 else ranked_items[0][1]
        runner_up_score = ranked_items[2][1] if round_idx == 0 and len(ranked_items) > 2 else (ranked_items[1][1] if len(ranked_items) > 1 else baseline_score)
        confidence_gap = baseline_score - runner_up_score

        path_rec = target_id
        if planned_path:
            if path_cursor < len(planned_path) - 1:
                path_rec = planned_path[path_cursor]
            elif planned_path:
                path_rec = planned_path[-1]
        if path_rec not in self.impressions:
            return baseline_rec

        path_score = self._score(path_rec, target_id)
        if confidence_gap >= self.baseline_confidence_threshold:
            return baseline_rec
        if path_score + self.path_override_margin < baseline_score:
            return baseline_rec
        return path_rec

    def _build_llm_candidates(self, user_vec: np.ndarray, target_id: str) -> List[str]:
        if target_id not in self.news:
            return []
        target_vec = self.news[target_id]
        scored = []
        for item_id in self.impressions:
            if item_id == target_id or item_id not in self.news:
                continue
            item_vec = self.news[item_id]
            sim_target = float(np.dot(item_vec, target_vec))
            sim_user = float(np.dot(item_vec, user_vec))
            score = sim_user + self.llm_target_prior * sim_target
            scored.append((item_id, score))
        scored.sort(key=lambda pair: -pair[1])
        return [item_id for item_id, _ in scored[:self.llm_candidate_k]]

    def _should_release_target(self, target_id: str, round_idx: int, planned_path: Optional[List[str]], path_cursor: int) -> bool:
        if not self.use_llm_planner or round_idx <= 0 or target_id not in self.impressions:
            return False
        ranked_items = self._rank_baseline_candidates(target_id)
        target_position = next((idx for idx, (item_id, _) in enumerate(ranked_items) if item_id == target_id), None)
        if target_position is not None and target_position + 1 <= self.llm_release_rank:
            return True
        if not planned_path:
            return False
        if path_cursor >= len(planned_path) - 1:
            return True
        next_path_item = planned_path[path_cursor]
        if next_path_item == target_id:
            return True
        target_score = self._score(target_id, target_id)
        next_path_score = self._score(next_path_item, target_id)
        return target_score + self.llm_safe_margin >= next_path_score and path_cursor >= max(len(planned_path) - 2, 0)

    def _build_reflection_rules(self, reflection: Optional[dict]) -> str:
        return build_reflection_rules(reflection, has_failed_paths=bool(self.failed_paths))

    def _build_llm_prompt(
        self,
        user_profile: str,
        history_ids: List[str],
        target_id: str,
        candidate_items: List[str],
        candidate_paths: List[List[str]],
        reflection: Optional[dict] = None,
    ) -> str:
        history_titles = '\n'.join(f'- {self._item_text(item_id)}' for item_id in history_ids[-10:]) or '- N/A'
        if isinstance(reflection, dict):
            last_action = reflection.get('last_action', 'none')
            outcome = reflection.get('outcome', 'none')
            failure_reason = reflection.get('reason_guess', 'unknown')
            rec_sim = float(reflection.get('rec_sim', 0.0))
            target_sim = float(reflection.get('target_sim', 0.0))
            reflection_triplet_text = (
                f'(state_deviation={float(reflection.get("state_deviation", 0.0)):.4f}, '
                f'target_contribution={float(reflection.get("target_contribution", 0.0)):.4f}, '
                f'feedback_label={int(reflection.get("feedback_label", 0))})'
            )
        else:
            last_action = 'none'
            outcome = 'none'
            failure_reason = 'none'
            rec_sim = 0.0
            target_sim = 0.0
            reflection_triplet_text = 'none'
        failed_path_lines = '\n'.join(' -> '.join(path) for path in self.failed_paths[-5:]) or 'none'
        candidate_path_lines = self._format_candidate_paths(candidate_paths, target_id)
        candidate_item_lines = '\n'.join(f'- {self._item_text(item_id)}' for item_id in candidate_items) or '- N/A'
        graph_topology_text = self._build_graph_topology_summary(target_id, candidate_items)
        return build_target_oriented_planner_prompt(
            history_titles=history_titles,
            user_profile=user_profile,
            target_text=self._item_text(target_id),
            candidate_item_lines=candidate_item_lines,
            last_action=last_action,
            outcome=outcome,
            failure_reason=failure_reason,
            rec_sim=rec_sim,
            target_sim=target_sim,
            policy_bias=self.policy_bias,
            reflection=reflection if isinstance(reflection, dict) else None,
            llm_path_k=self.llm_path_k,
            reject_streak=self.reject_streak,
            failed_path_lines=failed_path_lines,
            candidate_path_lines=candidate_path_lines,
            graph_topology_text=graph_topology_text,
            reflection_triplet_text=reflection_triplet_text,
            target_id=target_id,
        )

    def _get_reflection_preferences(self, reflection: Optional[dict]) -> dict:
        preferences = {
            'prefer_history': 0.0,
            'prefer_target': 0.0,
            'prefer_bridge': 0.0,
            'must_change_topic': False,
            'min_history_similarity': None,
        }
        if not isinstance(reflection, dict):
            return preferences
        outcome = reflection.get('outcome')
        reason = reflection.get('reason_guess')
        rec_sim = float(reflection.get('rec_sim', 0.0))
        target_sim = float(reflection.get('target_sim', 0.0))
        if outcome == 'reject':
            if reason == 'low_similarity':
                preferences['prefer_history'] = 0.45
                preferences['prefer_bridge'] = 0.10
                preferences['min_history_similarity'] = max(0.45, rec_sim + 0.03)
            elif reason == 'wrong_topic':
                preferences['must_change_topic'] = True
                preferences['prefer_target'] = 0.20
                preferences['prefer_history'] = 0.10
            elif reason == 'too_far':
                preferences['prefer_target'] = 0.35
                preferences['prefer_bridge'] = 0.15
                preferences['min_history_similarity'] = max(0.35, min(rec_sim, target_sim))
        elif outcome == 'accept':
            preferences['prefer_target'] = 0.20
            preferences['prefer_history'] = 0.10
        return preferences

    def _get_current_stage(self) -> str:
        return 'soft_bias'

    def _remember_failed_path(self, path: Optional[List[str]]):
        if not path:
            return
        normalized = tuple(str(item_id).strip().upper() for item_id in path if str(item_id).strip())
        if len(normalized) <= 1:
            return
        if normalized in [tuple(saved) for saved in self.failed_paths]:
            return
        self.failed_paths.append(list(normalized))
        self.failed_paths = self.failed_paths[-5:]

    def _path_is_previously_failed(self, path: List[str]) -> bool:
        normalized = tuple(str(item_id).strip().upper() for item_id in path if str(item_id).strip())
        return normalized in [tuple(saved) for saved in self.failed_paths]

    def _path_has_failed_structure(self, path: List[str]) -> bool:
        normalized = [str(item_id).strip().upper() for item_id in path if str(item_id).strip()]
        for failed_path in self.failed_paths:
            if len(normalized) >= 2 and len(failed_path) >= 2 and normalized[:2] == failed_path[:2]:
                return True
        return False

    def _path_target_sims(self, path: List[str], target_id: str) -> List[float]:
        if target_id not in self.news:
            return []
        target_vec = self.news[target_id]
        sims = []
        for item_id in path:
            if item_id not in self.news:
                return []
            sims.append(self._cosine(self.news[item_id], target_vec))
        return sims

    def _is_target_progressive_path(self, path: List[str], target_id: str) -> bool:
        sims = self._path_target_sims(path, target_id)
        if not sims:
            return False
        for idx in range(len(sims) - 1):
            if sims[idx + 1] <= sims[idx] + self.path_min_progress:
                return False
        return True

    def _plan_path_from_start(
        self,
        start_id: str,
        target_id: str,
        candidate_items: List[str],
        reflection: Optional[dict] = None,
    ) -> List[str]:
        if start_id not in self.news or target_id not in self.news:
            return [target_id]
        allowed_items = set(candidate_items)
        allowed_items.add(target_id)
        path = [start_id]
        visited = {start_id}
        current = start_id
        target_vec = self.news[target_id]
        history_anchor = self._compute_temporal_user_state(self.history)
        current_weights = dict(self.policy_bias)
        preferences = self._get_reflection_preferences(reflection)
        current_weights['history_weight'] += preferences['prefer_history']
        current_weights['target_weight'] += preferences['prefer_target']
        current_weights['bridge_weight'] += preferences['prefer_bridge']
        rejected_item = reflection.get('last_action') if isinstance(reflection, dict) else None
        rejected_category = self.cat.get(rejected_item, '') if rejected_item else ''
        for step_idx in range(max(self.max_path_length - 2, 0)):
            if current == target_id:
                break
            current_target_sim = self._cosine(self.news[current], target_vec)
            neighbors = [
                neighbor for neighbor in self.item_graph.get(current, [])
                if neighbor in allowed_items and neighbor not in visited
            ]
            if not neighbors:
                break
            ranked_neighbors = []
            for neighbor in neighbors:
                neighbor_vec = self.news[neighbor]
                target_sim = self._cosine(neighbor_vec, target_vec)
                history_sim = self._cosine(neighbor_vec, history_anchor)
                bridge_sim = self._cosine(self.news[current], neighbor_vec)
                noise = float(self.rng.normal(0, 0.015))
                base_score = self._score(neighbor, target_id)
                progress_gain = target_sim - current_target_sim
                score = (
                    base_score
                    + 0.12 * current_weights['bridge_weight'] * bridge_sim
                    + 0.04 * current_weights['history_weight'] * history_sim
                    + 0.04 * current_weights['target_weight'] * target_sim
                    + self.path_progress_weight * max(progress_gain, 0.0)
                    + noise
                )
                if step_idx == 0:
                    score += self.path_first_step_weight * base_score
                if neighbor != target_id and progress_gain <= self.path_min_progress:
                    score -= 3.0
                if neighbor == target_id and step_idx == 0 and history_sim < 0.55:
                    score -= 0.35
                if preferences['min_history_similarity'] is not None and history_sim < preferences['min_history_similarity']:
                    score -= 0.80
                if preferences['must_change_topic'] and rejected_category and self.cat.get(neighbor, '') == rejected_category:
                    score -= 1.25
                if rejected_item and neighbor == rejected_item:
                    score -= 1.50
                ranked_neighbors.append((neighbor, score))
            ranked_neighbors.sort(key=lambda pair: -pair[1])
            current = ranked_neighbors[0][0]
            path.append(current)
            visited.add(current)
        if path[-1] != target_id:
            path.append(target_id)
        return self._sanitize_llm_path(path, target_id, candidate_items)

    def _build_diverse_llm_paths(
        self,
        user_vec: np.ndarray,
        target_id: str,
        candidate_items: List[str],
        reflection: Optional[dict] = None,
        path_count: int = 3,
    ) -> List[List[str]]:
        if not candidate_items:
            return [[target_id]]
        target_vec = self.news[target_id]
        start_scores = []
        preferences = self._get_reflection_preferences(reflection)
        for item_id in candidate_items:
            if item_id not in self.news:
                continue
            item_vec = self.news[item_id]
            history_sim = self._cosine(item_vec, user_vec)
            target_sim = self._cosine(item_vec, target_vec)
            score = history_sim * (1.0 + preferences['prefer_history'])
            score += self.llm_target_prior * target_sim * (1.0 + preferences['prefer_target'])
            if preferences['min_history_similarity'] is not None and history_sim < preferences['min_history_similarity']:
                score -= 0.75
            if preferences['must_change_topic'] and isinstance(reflection, dict):
                last_action = reflection.get('last_action')
                if last_action and self.cat.get(item_id, '') == self.cat.get(last_action, ''):
                    score -= 1.0
            score += float(self.rng.normal(0, 0.03))
            start_scores.append((item_id, score))
        start_scores.sort(key=lambda pair: -pair[1])
        candidate_starts = [item_id for item_id, _ in start_scores[:max(path_count * 3, path_count)]]
        diverse_paths = []
        used_keys = set()
        for start_id in candidate_starts:
            path = self._plan_path_from_start(start_id, target_id, candidate_items, reflection)
            key = tuple(path)
            if not path or key in used_keys or self._path_is_previously_failed(path) or self._path_has_failed_structure(path):
                continue
            diverse_paths.append(path)
            used_keys.add(key)
            if len(diverse_paths) >= path_count:
                break
        if not diverse_paths:
            diverse_paths.append(self._sanitize_llm_path(self._plan_path(user_vec, target_id), target_id, candidate_items))
        return diverse_paths[:path_count]

    def _cosine(self, left_vec: np.ndarray, right_vec: np.ndarray) -> float:
        left_norm = float(np.linalg.norm(left_vec))
        right_norm = float(np.linalg.norm(right_vec))
        if left_norm < 1e-8 or right_norm < 1e-8:
            return 0.0
        return float(np.dot(left_vec, right_vec) / (left_norm * right_norm + 1e-8))

    def _score_path(self, path: List[str], user_vec: np.ndarray, target_vec: np.ndarray, alpha: float) -> float:
        if not path:
            return float('-inf')
        for item_id in path:
            if item_id not in self.news:
                return float('-inf')
        target_id = path[-1]
        metrics = self._path_structure_metrics(path, target_id, user_vec=user_vec)
        path_score = (
            0.35 * metrics['semantic_coherence']
            + alpha * metrics['history_consistency']
            + (1 - alpha) * metrics['target_proximity']
        )
        final_step_id = path[-2] if len(path) >= 2 else path[-1]
        final_sim = self._cosine(self.news[final_step_id], target_vec)
        first_step_bonus = 0.0
        if path and path[0] != path[-1]:
            first_step_bonus = self.path_first_step_weight * self._score(path[0], path[-1])
        return (
            path_score
            + (1 - alpha) * final_sim
            + self.path_progress_weight * metrics['mean_progress']
            + first_step_bonus
            - self.path_length_penalty * max(len(path) - 1, 0)
            - self.path_monotonic_penalty * metrics['monotonic_violations']
        )

    def _parse_path(self, response: str, candidate_items: Optional[List[str]] = None, target_id: Optional[str] = None) -> List[str]:
        cleaned = response.strip()
        if not cleaned:
            return []
        cleaned = cleaned.replace('```json', '').replace('```', '').strip()
        cleaned = cleaned.replace('→', '->').replace('=>', '->').replace('➡', '->')
        candidate_items = candidate_items or []
        candidate_pool = list(dict.fromkeys(candidate_items + ([target_id] if target_id else [])))

        def normalize_items(raw_items: List[str]) -> List[str]:
            normalized = []
            for raw_item in raw_items:
                text = str(raw_item).strip().strip('"\'[]()')
                if not text:
                    continue
                if re.fullmatch(r'\d+', text) and candidate_pool:
                    idx = int(text)
                    if 0 <= idx < len(candidate_pool):
                        normalized.append(candidate_pool[idx])
                    continue
                match = re.search(r'\bN\d+\b', text, flags=re.IGNORECASE)
                if match:
                    normalized.append(match.group(0).upper())
            deduped = []
            seen = set()
            for item_id in normalized:
                if item_id not in seen:
                    deduped.append(item_id)
                    seen.add(item_id)
            return deduped

        bracket_match = re.search(r'\[[\s\S]*?\]', cleaned)
        if bracket_match:
            try:
                parsed = json.loads(bracket_match.group(0))
                if isinstance(parsed, list):
                    return normalize_items([str(item) for item in parsed])
            except json.JSONDecodeError:
                pass
        regex_items = re.findall(r'\bN\d+\b', cleaned, flags=re.IGNORECASE)
        if regex_items:
            return normalize_items(regex_items)
        if cleaned.startswith('[') and cleaned.endswith(']'):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    return normalize_items([str(item) for item in parsed])
            except json.JSONDecodeError:
                pass
        items = [item.strip().strip('"\'[]()') for item in cleaned.split('->') if item.strip()]
        return normalize_items(items)

    def _parse_path_candidates(self, response: str, candidate_items: Optional[List[str]] = None, target_id: Optional[str] = None) -> Dict[str, List[List[str]]]:
        cleaned = response.strip().replace('```json', '').replace('```', '').strip()
        best_path = []
        candidate_paths = []
        reason = ''
        response_stage = ''

        def add_path(path: List[str]):
            if path:
                candidate_paths.append(path)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                reason = str(parsed.get('reason', '')).strip()
                response_stage = str(parsed.get('stage', '')).strip().lower()
                best_raw = parsed.get('best_path') or parsed.get('selected_path') or parsed.get('chosen_path') or parsed.get('final_path')
                if isinstance(best_raw, list):
                    best_path = self._sanitize_llm_path(self._parse_path(json.dumps(best_raw), candidate_items, target_id), target_id, candidate_items or [])
                raw_candidates = parsed.get('candidates') or parsed.get('candidate_paths') or parsed.get('paths') or parsed.get('alternatives') or []
                if isinstance(raw_candidates, list):
                    for raw_path in raw_candidates:
                        if isinstance(raw_path, list):
                            add_path(self._sanitize_llm_path(self._parse_path(json.dumps(raw_path), candidate_items, target_id), target_id, candidate_items or []))
            elif isinstance(parsed, list):
                if parsed and all(isinstance(item, list) for item in parsed):
                    for raw_path in parsed:
                        add_path(self._sanitize_llm_path(self._parse_path(json.dumps(raw_path), candidate_items, target_id), target_id, candidate_items or []))
                else:
                    best_path = self._sanitize_llm_path(self._parse_path(json.dumps(parsed), candidate_items, target_id), target_id, candidate_items or [])
        except json.JSONDecodeError:
            pass

        if not best_path:
            best_path = self._sanitize_llm_path(self._parse_path(response, candidate_items, target_id), target_id, candidate_items or [])
        if best_path:
            candidate_paths.insert(0, best_path)
        deduped_paths = []
        seen = set()
        for path in candidate_paths:
            if not path:
                continue
            key = tuple(path)
            if key not in seen:
                deduped_paths.append(path)
                seen.add(key)
        return {'best_path': best_path, 'candidate_paths': deduped_paths, 'reason': reason, 'stage': response_stage}

    def _sanitize_llm_path(self, path: List[str], target_id: str, candidate_items: List[str]) -> List[str]:
        candidate_set = set(candidate_items)
        sanitized = []
        seen = set()
        for item_id in path:
            normalized = str(item_id).strip().upper()
            if normalized == target_id:
                continue
            if normalized in candidate_set and normalized not in seen:
                sanitized.append(normalized)
                seen.add(normalized)
        if target_id and (not sanitized or sanitized[-1] != target_id):
            sanitized.append(target_id)
        return sanitized

    def _validate_path(self, path: List[str], target_id: str, candidate_items: List[str]) -> bool:
        if not path or path[-1] != target_id:
            return False
        candidate_set = set(candidate_items)
        for item_id in path:
            if item_id != target_id and item_id not in candidate_set:
                return False
        return True

    def _self_reflect(self, rec: str, target_id: str, accept: bool, previous_user_vec: Optional[np.ndarray] = None) -> dict:
        base_user = self.user if previous_user_vec is None else previous_user_vec
        target_sim_before = self._target_similarity(target_id, base_user)
        target_sim_after = self._target_similarity(target_id)
        rec_sim = float(np.dot(base_user, self.news[rec]))
        target_sim = target_sim_after
        reason_guess = 'aligned'
        if not accept:
            if target_sim - rec_sim > 0.12:
                reason_guess = 'too_far'
            elif rec_sim < 0.55:
                reason_guess = 'low_similarity'
            else:
                reason_guess = 'wrong_topic'
        state_deviation = float(max(0.0, 1.0 - rec_sim))
        target_contribution = float(target_sim_after - target_sim_before)
        reflection = {
            'last_action': rec,
            'outcome': 'accept' if accept else 'reject',
            'reason_guess': reason_guess,
            'target_gap': float(target_sim - rec_sim),
            'rec_sim': rec_sim,
            'target_sim': target_sim,
            'state_deviation': state_deviation,
            'target_contribution': target_contribution,
            'feedback_label': 1 if accept else 0,
            'reflection_triplet': (state_deviation, target_contribution, 1 if accept else 0),
        }
        self._update_policy(reflection)
        return reflection

    def _update_policy(self, reflection: dict):
        outcome = reflection.get('outcome')
        reason_guess = reflection.get('reason_guess')
        rec_sim = float(reflection.get('rec_sim', 0.0))
        target_sim = float(reflection.get('target_sim', 0.0))
        target_gap = max(float(reflection.get('target_gap', 0.0)), 0.0)
        if outcome == 'reject':
            bridge_delta = 0.04 + min(target_gap, 0.3) * 0.35
            target_delta = 0.03 + max(0.0, 0.55 - rec_sim) * 0.25
            history_delta = 0.02 + max(0.0, rec_sim - target_sim) * 0.15
            if reason_guess == 'too_far':
                self.policy_bias['target_weight'] += 0.10 + bridge_delta
                self.policy_bias['bridge_weight'] += 0.05
            elif reason_guess == 'low_similarity':
                self.policy_bias['history_weight'] += 0.12 + target_delta
                self.policy_bias['bridge_weight'] += 0.04
            else:
                self.policy_bias['target_weight'] += 0.08
                self.policy_bias['history_weight'] += 0.04
                self.policy_bias['bridge_weight'] += 0.10 + history_delta
        elif outcome == 'accept':
            accept_bonus = 0.03 + max(0.0, rec_sim - 0.5) * 0.1
            self.policy_bias['history_weight'] += accept_bonus
            self.policy_bias['target_weight'] += 0.08 + max(0.0, target_sim - rec_sim) * 0.08
        for key in self.policy_bias:
            self.policy_bias[key] = float(np.clip(self.policy_bias[key], 0.5, 2.0))

    def _select_local_rescue_path(self, candidate_paths: List[List[str]], user_vec: np.ndarray, target_id: str, candidate_items: List[str]) -> List[str]:
        if target_id not in self.news:
            return []
        target_vec = self.news[target_id]
        valid_paths = [
            path for path in candidate_paths
            if path and self._validate_path(path, target_id, candidate_items) and not self._path_is_previously_failed(path)
        ]
        strict_paths = [path for path in valid_paths if self._is_target_progressive_path(path, target_id)]
        if strict_paths:
            return max(strict_paths, key=lambda path: self._score_path(path, user_vec, target_vec, alpha=0.45))
        if valid_paths:
            return max(valid_paths, key=lambda path: self._score_path(path, user_vec, target_vec, alpha=0.45))
        sanitized_paths = [self._sanitize_llm_path(path, target_id, candidate_items) for path in candidate_paths if path]
        sanitized_paths = [path for path in sanitized_paths if path]
        strict_sanitized_paths = [path for path in sanitized_paths if self._is_target_progressive_path(path, target_id)]
        if strict_sanitized_paths:
            return max(strict_sanitized_paths, key=lambda path: self._score_path(path, user_vec, target_vec, alpha=0.45))
        if sanitized_paths:
            return max(sanitized_paths, key=lambda path: self._score_path(path, user_vec, target_vec, alpha=0.45))
        return []

    def _llm_plan_path(self, user_vec: np.ndarray, target_id: str, reflection: Optional[dict] = None) -> List[str]:
        self.llm_called += 1
        if self.debug:
            print(f'  llm planner activated: call={self.llm_called}, target={target_id}')
        candidate_items = self._build_llm_candidates(user_vec, target_id)
        self._refresh_intention_graph(target_id, candidate_items)
        candidate_paths = self._build_diverse_llm_paths(user_vec, target_id, candidate_items, reflection=reflection, path_count=3)
        user_profile = self._build_user_profile()
        prompt = self._build_llm_prompt(user_profile, self.history, target_id, candidate_items, candidate_paths, reflection)
        main_idea_path = self._plan_path(user_vec, target_id)
        rescue_path = self._select_local_rescue_path(candidate_paths, user_vec, target_id, candidate_items) if self.enable_llm_rescue else []
        remote_paths = []
        try:
            response = self._get_planner_llm().plan_path(prompt)
            self.llm_api_success += 1
            if self.debug:
                print(f'  llm raw response: {response}')
            parsed_bundle = self._parse_path_candidates(response, candidate_items, target_id)
            remote_paths = list(parsed_bundle.get('candidate_paths', [])) if 'parsed_bundle' in locals() else []
        except Exception as exc:
            self.llm_api_failure += 1
            error_message = str(exc).strip()
            if error_message and len(self.llm_api_error_messages) < 20:
                self.llm_api_error_messages.append(error_message)
            if rescue_path:
                self.llm_success += 1
                self.llm_rescue_adopted += 1
                self.llm_path_lengths.append(len(rescue_path))
                self.llm_path_length_sum += len(rescue_path)
                return rescue_path
            self.llm_fallback += 1
            if self.debug:
                print(f'  llm fallback to greedy: {error_message}')
            return self._plan_path(user_vec, target_id)
        self.last_llm_reason = str(parsed_bundle.get('reason', '')).strip()
        self.last_llm_stage = 'soft_bias'
        paths = parsed_bundle.get('candidate_paths', []) if 'parsed_bundle' in locals() else []
        for candidate_path in candidate_paths:
            if candidate_path and tuple(candidate_path) not in [tuple(path) for path in paths]:
                paths.append(candidate_path)
        valid_paths = [path for path in paths if self._validate_path(path, target_id, candidate_items)]
        if not paths or not any(path for path in paths):
            self.llm_invalid_path += 1
            if rescue_path:
                self.llm_success += 1
                self.llm_rescue_adopted += 1
                self.llm_path_lengths.append(len(rescue_path))
                self.llm_path_length_sum += len(rescue_path)
                return rescue_path
            self.llm_fallback += 1
            return self._plan_path(user_vec, target_id)
        if not valid_paths:
            self.llm_invalid_path += 1
            valid_paths = [self._sanitize_llm_path(path, target_id, candidate_items) for path in paths if path]
            valid_paths = [path for path in valid_paths if path]
        if not valid_paths:
            if rescue_path:
                self.llm_success += 1
                self.llm_rescue_adopted += 1
                self.llm_path_lengths.append(len(rescue_path))
                self.llm_path_length_sum += len(rescue_path)
                return rescue_path
            self.llm_fallback += 1
            return self._plan_path(user_vec, target_id)
        best_path = parsed_bundle.get('best_path', []) if 'parsed_bundle' in locals() else []
        progressive_valid_paths = [path for path in valid_paths if self._is_target_progressive_path(path, target_id)]
        if (
            not self._validate_path(best_path, target_id, candidate_items)
            or self._path_is_previously_failed(best_path)
            or not self._is_target_progressive_path(best_path, target_id)
        ):
            candidate_pool = progressive_valid_paths or valid_paths
            best_path = next((path for path in candidate_pool if not self._path_is_previously_failed(path)), candidate_pool[0])
        target_vec = self.news[target_id]
        llm_path_score = self._score_path(best_path, user_vec, target_vec, alpha=0.45)
        main_idea_score = self._score_path(main_idea_path, user_vec, target_vec, alpha=0.45)
        remote_path_keys = {tuple(path) for path in remote_paths if path}
        adopted_key = tuple(best_path) if best_path else tuple()
        if self.enable_llm_guard and main_idea_path and llm_path_score < main_idea_score + self.llm_safe_margin:
            best_path = main_idea_path
            self.llm_guard_adopted += 1
        elif adopted_key in remote_path_keys:
            self.llm_remote_adopted += 1
        else:
            self.llm_guard_adopted += 1
        self.llm_success += 1
        self.llm_path_lengths.append(len(best_path))
        self.llm_path_length_sum += len(best_path)
        return best_path

    def _sample_target(self) -> Optional[str]:
        if not self.impressions:
            return None
        sims = [(item_id, float(np.dot(self.initial_user, self.news[item_id]))) for item_id in self.impressions]
        sims.sort(key=lambda x: x[1])
        low = int(len(sims) * self.target_percentile_low)
        high = int(len(sims) * self.target_percentile_high)
        low = min(max(low, 0), max(len(sims) - 1, 0))
        high = min(len(sims), max(low + 1, high))
        candidates = [item_id for item_id, _ in sims[low:high]]
        if not candidates:
            return sims[min(len(sims) - 1, len(sims) // 3)][0]
        return candidates[self.rng.integers(len(candidates))]

    def _score(self, item_id: str, target_id: str) -> float:
        item_vec = self.news[item_id]
        target_vec = self.news[target_id]
        i_val = self._sigmoid(np.dot(self.user, item_vec))
        direction = self._normalize(item_vec - self.user)
        g_val = self._sigmoid(np.dot(target_vec, direction))
        alpha = self._compute_openness_fine_grained(item_id) if self.alpha_fn else self.alpha
        i_val = np.clip(i_val, self.eps, 1 - self.eps)
        g_val = np.clip(g_val, self.eps, 1 - self.eps)
        score = (i_val ** (1 - alpha)) * (g_val ** alpha)
        return float(score + self.rng.normal(0, 0.01))

    def _feedback(self, item_id: str) -> bool:
        sim = float(np.dot(self.user, self.news[item_id]))
        prob = self._sigmoid(self.feedback_scale * sim - self.feedback_bias)
        return bool(self.rng.random() < prob)

    def _get_planner_llm(self):
        if self.planner_llm is None:
            self.planner_llm = get_planner_engine()
        return self.planner_llm

    def _get_feedback_llm(self):
        return self._get_planner_llm()

    def _llm_feedback(self, item_id: str) -> bool:
        self.llm_feedback_calls += 1
        cache_key = (tuple(self.history[-5:]), item_id)
        if cache_key in self.llm_feedback_cache:
            self.llm_feedback_cache_hits += 1
            return self.llm_feedback_cache[cache_key]
        candidate_title = self.item_titles.get(item_id, item_id)
        history_titles = [self.item_titles.get(history_id, history_id) for history_id in self.history[-5:] if history_id in self.item_titles]
        engine = self._get_feedback_llm()
        prompt = engine.build_prompt(history_titles, candidate_title)
        output = engine.generate(prompt)
        normalized_output = output.strip().lower()
        if 'yes' in normalized_output:
            result = True
        elif 'no' in normalized_output:
            result = False
        else:
            self.llm_feedback_parse_failures += 1
            result = False
        self.llm_feedback_cache[cache_key] = result
        return result

    def _update_user(self, item_id: str):
        vec = self.news[item_id]
        temporal_state = self._compute_temporal_user_state(self.history + [item_id])
        self.user = (1 - self.user_update_eta) * self.user + self.user_update_eta * vec
        self.user = (1 - self.temporal_state_mix) * self.user + self.temporal_state_mix * temporal_state
        self.user = self.user / (np.linalg.norm(self.user) + 1e-8)

    def _state_transition(self, item_id: str, feedback: bool) -> str:
        if feedback:
            self._update_user(item_id)
            self.history.append(item_id)
            return 'accept_update'
        if self.reject_decay > 0.0:
            self.user = self._normalize((1.0 - self.reject_decay) * self.user + self.reject_decay * self.initial_user)
            return 'reject_decay'
        return 'reject_hold'

    def _target_similarity(self, target_id: str, user_vec: Optional[np.ndarray] = None) -> float:
        base_user = self.user if user_vec is None else user_vec
        return float(np.dot(base_user, self.news[target_id]))

    def _target_rank(self, target_id: str, user_vec: Optional[np.ndarray] = None) -> int:
        base_user = self.user if user_vec is None else user_vec
        scores = [float(np.dot(base_user, self.news[item_id])) for item_id in self.impressions]
        target_score = float(np.dot(base_user, self.news[target_id]))
        return int(sum(score > target_score for score in scores) + 1)

    def run(self, max_rounds: int = 20) -> list:
        if not self.impressions:
            return []
        target = self._sample_target()
        if target is None:
            return []
        planned_path = None
        path_cursor = 0
        llm_phase_switched = False
        if self.use_path_planner:
            planned_path = self._llm_plan_path(self.user.copy(), target) if self.use_llm_planner else self._plan_path(self.user.copy(), target)
        trajectory = []
        for round_idx in range(max_rounds):
            pre_action_user = self.user.copy()
            if self.use_path_planner and planned_path:
                if self._should_release_target(target, round_idx, planned_path, path_cursor):
                    rec = target
                elif self.use_llm_planner and round_idx >= self.llm_hybrid_rounds:
                    if not llm_phase_switched:
                        planned_path = self._plan_path(self.user.copy(), target)
                        path_cursor = 0
                        llm_phase_switched = True
                    if self.enable_gating:
                        rec = self._select_baseline_first_rec(round_idx, target, planned_path, path_cursor)
                    elif path_cursor < len(planned_path) - 1:
                        rec = planned_path[path_cursor]
                    else:
                        rec = target
                elif not self.use_llm_planner and self.enable_gating:
                    rec = self._select_baseline_first_rec(round_idx, target, planned_path, path_cursor)
                elif not self.use_llm_planner and path_cursor < len(planned_path) - 1:
                    rec = planned_path[path_cursor]
                elif path_cursor < len(planned_path) - 1:
                    rec = planned_path[path_cursor]
                else:
                    rec = target
                if round_idx == 0 and rec == target:
                    non_target_steps = [item_id for item_id in planned_path if item_id != target]
                    if non_target_steps:
                        rec = non_target_steps[0]
                    else:
                        fallback_scores = {i: self._score(i, target) for i in self.impressions if i != target}
                        if fallback_scores:
                            rec = max(fallback_scores, key=fallback_scores.get)
                if rec not in self.impressions:
                    scores = {i: self._score(i, target) for i in self.impressions}
                    rec = max(scores, key=scores.get)
                elif path_cursor < len(planned_path) - 1:
                    path_cursor += 1
            else:
                scores = {i: self._score(i, target) for i in self.impressions}
                if round_idx == 0:
                    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
                    rec = sorted_items[1][0]
                else:
                    rec = max(scores, key=scores.get)
            rec_alpha = self._compute_openness_fine_grained(rec) if self.alpha_fn else self.alpha
            accept = self._feedback(rec)
            success = accept and (rec == target)
            self._state_transition(rec, accept)
            if accept:
                self.reject_streak = 0
                if self.use_path_planner and not success and self.enable_replanning:
                    reflection = self._self_reflect(rec, target, accept=True, previous_user_vec=pre_action_user) if self.use_llm_planner and self.enable_reflection else None
                    planned_path = self._llm_plan_path(self.user.copy(), target, reflection=reflection) if self.use_llm_planner else self._plan_path(self.user.copy(), target)
                    path_cursor = 0
            if (not accept) and self.use_path_planner and not self.use_llm_planner and not success and self.enable_replanning:
                planned_path = self._plan_path(self.user.copy(), target)
                path_cursor = 0
            if (not accept) and self.use_llm_planner and not success:
                self.reject_streak += 1
                self._remember_failed_path(planned_path)
                if self.enable_replanning:
                    reflection = self._self_reflect(rec, target, accept=False, previous_user_vec=pre_action_user) if self.enable_reflection else None
                    planned_path = self._llm_plan_path(self.user.copy(), target, reflection=reflection)
                    path_cursor = 0
            trajectory.append({
                'round': round_idx + 1,
                'rec': rec,
                'target': target,
                'alpha': rec_alpha,
                'accept': accept,
                'success': success,
                'target_rating': self._target_similarity(target),
                'target_rank': self._target_rank(target),
                'path_step': round_idx + 1 if self.use_path_planner else None,
            })
        return trajectory
