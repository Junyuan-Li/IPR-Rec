from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.runtime_utils import stable_seed_from_text


class NativeONeRecBaseline:
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
        planner_mode: str = 'baseline',
        graph_k: int = 10,
        max_path_length: int = 5,
        llm_candidate_k: int = 20,
        llm_path_k: int = 5,
        llm_target_prior: float = 0.65,
        llm_hybrid_rounds: int = 2,
        enable_gating: bool = True,
        enable_replanning: bool = True,
        enable_reflection: bool = True,
    ):
        del feedback_threshold
        del planner_mode
        del graph_k
        del max_path_length
        del llm_candidate_k
        del llm_path_k
        del llm_target_prior
        del llm_hybrid_rounds
        del enable_gating
        del enable_replanning
        del enable_reflection

        self.news = news_embeds
        self.cat = categories
        self.history = user_history.copy()
        self.item_titles = item_titles or {}
        self.reach_min = reach_min_sim
        self.debug = debug
        self.impressions = [item_id for item_id in impressions if item_id in news_embeds]

        session_key = '|'.join(user_history) + '||' + '|'.join(impressions)
        self.random_seed = random_seed if random_seed is not None else stable_seed_from_text(session_key)
        self.rng = np.random.default_rng(self.random_seed)

        vecs = [self.news[item_id] for item_id in self.history if item_id in self.news]
        if vecs:
            user_vec = np.mean(vecs, axis=0)
            self.user = user_vec / (np.linalg.norm(user_vec) + 1e-8)
        else:
            dim = next(iter(self.news.values())).shape[0]
            self.user = np.zeros(dim)

        self.lambda0 = 0.1
        self.lambda1 = 0.2
        self.eps = 1e-8
        self.target_percentile_low = 0.2
        self.target_percentile_high = 0.45
        self.feedback_scale = 2.0
        self.feedback_bias = 1.0
        self.alpha_min = 0.1
        self.alpha_max = 0.55
        self.alpha_w1 = 0.5
        self.alpha_w2 = 0.5
        self.alpha_fn = True
        self.initial_user = self.user.copy()
        self.alpha = self._compute_openness()

    def _sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -10, 10)))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        return vector / (np.linalg.norm(vector) + 1e-8)

    def _compute_openness(self) -> float:
        if len(self.history) < 2:
            return self.lambda0

        transitions = sum(
            1
            for idx in range(len(self.history) - 1)
            if self.cat.get(self.history[idx], '') != self.cat.get(self.history[idx + 1], '')
        )
        frequency = transitions / (len(self.history) - 1)

        vecs = [self.news[item_id] for item_id in self.history if item_id in self.news]
        if len(vecs) >= 2:
            sims = cosine_similarity(vecs)
            upper = sims[np.triu_indices(len(vecs), 1)]
            diversity = float(1.0 - np.mean(upper))
        else:
            diversity = 0.0

        return float(np.clip(self.lambda0 + self.lambda1 * (frequency + diversity), self.alpha_min, self.alpha_max))

    def _compute_openness_fine_grained(self, item_id: str) -> float:
        item_vec = self.news[item_id]
        hist_vecs = [self.news[item] for item in self.history if item in self.news]
        user_centroid = np.mean(hist_vecs, axis=0) if hist_vecs else self.user

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

    def _sample_target(self) -> Optional[str]:
        if not self.impressions:
            return None

        sims = [(item_id, float(np.dot(self.initial_user, self.news[item_id]))) for item_id in self.impressions]
        sims.sort(key=lambda pair: pair[1])

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

    def _update_user(self, item_id: str):
        vec = self.news[item_id]
        diffs = []
        for idx in range(len(self.history) - 1):
            left_id = self.history[idx]
            right_id = self.history[idx + 1]
            if left_id in self.news and right_id in self.news:
                sim = cosine_similarity([self.news[left_id]], [self.news[right_id]])[0][0]
                diffs.append(1.0 - sim)
        omega = float(np.clip(np.mean(diffs), 0.1, 0.5)) if diffs else 0.1
        self.user = (1 - omega) * self.user + omega * vec
        self.user = self.user / (np.linalg.norm(self.user) + 1e-8)

    def _state_transition(self, item_id: str, feedback: bool) -> str:
        if feedback:
            self._update_user(item_id)
            self.history.append(item_id)
            return 'accept_update'
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

        trajectory = []
        for round_idx in range(max_rounds):
            scores = {item_id: self._score(item_id, target) for item_id in self.impressions}
            if round_idx == 0:
                sorted_items = sorted(scores.items(), key=lambda pair: -pair[1])
                rec = sorted_items[1][0] if len(sorted_items) > 1 else sorted_items[0][0]
            else:
                rec = max(scores, key=scores.get)

            rec_alpha = self._compute_openness_fine_grained(rec) if self.alpha_fn else self.alpha
            accept = self._feedback(rec)
            success = accept and (rec == target)
            self._state_transition(rec, accept)
            trajectory.append({
                'round': round_idx + 1,
                'rec': rec,
                'target': target,
                'alpha': rec_alpha,
                'accept': accept,
                'success': success,
                'target_rating': self._target_similarity(target),
                'target_rank': self._target_rank(target),
                'path_step': None,
            })

        return trajectory