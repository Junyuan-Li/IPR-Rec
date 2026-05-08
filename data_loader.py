import os
from collections import defaultdict

import numpy as np

from runtime_utils import stable_seed_from_text


class MINDDataLoader:
    """MIND 数据集加载器"""

    def __init__(self, data_dir: str, min_interactions: int = 5, max_history: int = 30):
        self.news = {}
        self.news_titles = {}
        self.users = {}
        self.categories = set()
        self.item_popularity = defaultdict(int)

        news_file = os.path.join(data_dir, 'news.tsv')
        behaviors_file = os.path.join(data_dir, 'behaviors.tsv')

        self._load_news(news_file)
        self._load_behaviors(behaviors_file, min_interactions, max_history)

    def _load_news(self, news_file: str):
        with open(news_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    self.news[parts[0]] = parts[1]
                    self.news_titles[parts[0]] = parts[3]
                    self.categories.add(parts[1])

    def _load_behaviors(self, behaviors_file: str, min_interactions: int, max_history: int):
        user_all_history = defaultdict(list)
        user_impressions = defaultdict(set)

        with open(behaviors_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                user_id = parts[1]
                history_str = parts[3]
                impressions_str = parts[4]

                history = history_str.split() if history_str.strip() else []
                impressions = [imp.split('-')[0] for imp in impressions_str.split() if imp]

                user_all_history[user_id].extend(history)
                user_impressions[user_id].update(impressions)

                for nid in history:
                    self.item_popularity[nid] += 1
                for nid in impressions:
                    self.item_popularity[nid] += 1

        for user_id, all_hist in user_all_history.items():
            seen = set()
            deduped = []
            for nid in reversed(all_hist):
                if nid not in seen:
                    seen.add(nid)
                    deduped.append(nid)
            deduped = deduped[::-1]
            hist = deduped[-max_history:] if deduped else []
            if len(hist) < min_interactions:
                continue
            self.users[user_id] = {
                'history': hist,
                'categories': [self.news.get(n, '') for n in hist if n in self.news],
                'impressions': sorted(user_impressions[user_id]),
            }

    def build_click_biased_impressions(
        self,
        user_id: str,
        all_items: list,
        size: int = 50,
        hist_ratio: float = 0.2,
        same_cat_ratio: float = 0.2,
        pop_ratio: float = 0.1,
        cross_cat_ratio: float = 0.5,
    ) -> list:
        seed = stable_seed_from_text(user_id)
        rng = np.random.default_rng(seed)
        ud = self.users.get(user_id)

        if ud is None:
            idx = rng.choice(len(all_items), size=min(size, len(all_items)), replace=False)
            return [all_items[i] for i in idx]

        history_list = ud['history']
        history_set = set(history_list)
        history_cats = set(c for c in ud.get('categories', []) if c)
        all_set = set(all_items)

        hist_in_all = [i for i in history_list if i in all_set]
        n_hist = int(size * hist_ratio)
        if len(hist_in_all) <= n_hist:
            hist_sample = hist_in_all
        else:
            hist_sample = rng.choice(hist_in_all, size=n_hist, replace=False).tolist()
        used = set(hist_sample)

        same_cat_pool = [
            nid for nid, cat in self.news.items()
            if cat in history_cats and nid not in history_set and nid in all_set
        ]
        n_same = int(size * same_cat_ratio)
        same_sample = rng.choice(
            same_cat_pool,
            size=min(n_same, len(same_cat_pool)),
            replace=False,
        ).tolist() if same_cat_pool else []
        used.update(same_sample)

        n_pop = int(size * pop_ratio)
        pop_candidates = sorted(
            [nid for nid in all_items if nid not in history_set],
            key=lambda nid: (-self.item_popularity.get(nid, 0), nid),
        )
        pop_cutoff = max(1, len(pop_candidates) * 3 // 10)
        pop_pool = pop_candidates[:pop_cutoff]
        pop_filtered = [i for i in pop_pool if i not in used]
        pop_sample = rng.choice(
            pop_filtered,
            size=min(n_pop, len(pop_filtered)),
            replace=False,
        ).tolist() if pop_filtered else []
        used.update(pop_sample)

        n_cross = size - len(hist_sample) - len(same_sample) - len(pop_sample)
        cross_pool = [
            nid for nid, cat in self.news.items()
            if cat not in history_cats and nid in all_set and nid not in used
        ]
        cross_sample = rng.choice(
            cross_pool,
            size=min(n_cross, len(cross_pool)),
            replace=False,
        ).tolist() if cross_pool else []

        combined = list(dict.fromkeys(hist_sample + same_sample + pop_sample + cross_sample))
        if len(combined) < size:
            remaining = [i for i in all_items if i not in set(combined)]
            if remaining:
                extra_n = min(size - len(combined), len(remaining))
                extra = rng.choice(remaining, size=extra_n, replace=False).tolist()
                combined.extend(extra)

        return combined[:size]

    def get_category_embedding(self, category: str) -> np.ndarray:
        cats = sorted(self.categories)
        emb = np.zeros(len(cats))
        try:
            emb[cats.index(category)] = 1.0
        except ValueError:
            pass
        return emb