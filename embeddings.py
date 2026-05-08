import os
import subprocess
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def build_news_embeddings(data_dir: str, data_loader, all_items, use_minilm: bool = True):
    news_embeds = {}
    embed_type = ''

    if use_minilm:
        try:
            from sentence_transformers import SentenceTransformer

            print('[*] 加载 all-MiniLM-L6-v2 ...')
            news_titles = {}
            news_file = os.path.join(data_dir, 'news.tsv')
            with open(news_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        news_titles[parts[0]] = parts[3]

            print('[*] 下载模型 all-MiniLM-L6-v2 (第一次运行会下载，~50MB)...')
            model = SentenceTransformer('all-MiniLM-L6-v2')
            ids = sorted(news_titles.keys())
            titles = [news_titles[i] for i in ids]
            print(f'[*] 编码 {len(ids)} 条新闻标题（batch_size=512）...')
            embeds = model.encode(
                titles,
                batch_size=512,
                show_progress_bar=True,
                normalize_embeddings=False,
            )
            embeds = embeds - embeds.mean(axis=0, keepdims=True)
            embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8)
            for nid, vec in zip(ids, embeds):
                news_embeds[nid] = vec.astype(np.float32)
            dim = embeds.shape[1]
            for item_id in all_items:
                if item_id not in news_embeds:
                    news_embeds[item_id] = np.zeros(dim, dtype=np.float32)
            embed_type = 'MiniLM-L6-v2 (论文指定)'
            print(f'[OK] MiniLM 嵌入完成: {len(news_embeds)} 条, 维度={dim}')
            return news_embeds, embed_type
        except ImportError as exc:
            print(f'[!] sentence-transformers 导入失败: {exc}')
            print('[*] 尝试自动安装...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'sentence-transformers'])
            return build_news_embeddings(data_dir, data_loader, all_items, use_minilm=True)
        except Exception as exc:
            print(f'[!] sentence-transformers 安装失败: {exc}')
            print('[!] 回退到 one-hot embedding')

    print('[*] 使用 one-hot category embedding...')
    for item_id in all_items:
        category = data_loader.news[item_id]
        news_embeds[item_id] = data_loader.get_category_embedding(category)
    dim = len(data_loader.categories)
    embed_type = f'one-hot category ({dim}维) [!] 非论文设定'
    print(f'[OK] one-hot 嵌入完成: {len(news_embeds)} 条, 维度={dim}')
    return news_embeds, embed_type


def diagnose_embedding_quality(news_embeds: dict) -> dict:
    sample_ids = sorted(news_embeds.keys())
    np.random.seed(42)
    if len(sample_ids) > 500:
        sample_ids = list(np.random.choice(sample_ids, 500, replace=False))
    sample_vecs = np.array([news_embeds[i] for i in sample_ids])
    cos_matrix = cosine_similarity(sample_vecs)
    upper_tri = cos_matrix[np.triu_indices(len(sample_vecs), k=1)]
    return {
        'cos_mean': float(np.mean(upper_tri)),
        'cos_std': float(np.std(upper_tri)),
    }