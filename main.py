#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from data_loader import MINDDataLoader
from embeddings import build_news_embeddings, diagnose_embedding_quality
from exp1_runner import DEFAULT_EXP1_CONFIG, run_exp1_pipeline
from runtime_utils import (
    GLOBAL_SEED,
    configure_stdout_utf8,
    install_required_packages,
    print_gpu_info,
    resolve_data_dir,
    set_global_seed,
    validate_data_files,
)


def main():
    configure_stdout_utf8()
    print('[*] ONeRec 初始化...')
    set_global_seed(GLOBAL_SEED)
    print(f'[*] 全局随机种子: {GLOBAL_SEED}')

    print('[*] 检查依赖包...')
    install_required_packages()
    print('[OK] 所有依赖已就绪')
    print_gpu_info()

    print('[*] 检测数据位置...')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = resolve_data_dir(script_dir)
    news_path, behaviors_path = validate_data_files(data_dir)
    print(f'[OK] 数据位置: {data_dir}')
    print(f'     news.tsv:      {os.path.getsize(news_path) / 1e6:.1f}MB')
    print(f'     behaviors.tsv: {os.path.getsize(behaviors_path) / 1e6:.1f}MB')

    print('[*] 加载 MIND 数据...')
    data_loader = MINDDataLoader(data_dir, min_interactions=5, max_history=30)
    print(f'[OK] 新闻项: {len(data_loader.news)}')
    print(f'[OK] 类别数: {len(data_loader.categories)}')
    print(f'[OK] 有效用户: {len(data_loader.users)}')

    print('[*] 构建新闻嵌入...')
    all_items = sorted(data_loader.news.keys())
    news_embeds, embed_type = build_news_embeddings(data_dir, data_loader, all_items, use_minilm=True)
    print(f'\n[嵌入类型] {embed_type}')
    print(f'[嵌入总数] {len(news_embeds)}')

    diag = diagnose_embedding_quality(news_embeds)
    print('\n' + '=' * 72)
    print('[DIAG] Embedding 质量检测')
    print('=' * 72)
    print(f"  pairwise cosine: mean={diag['cos_mean']:.4f}  std={diag['cos_std']:.4f}")
    if diag['cos_std'] > 0.1:
        print('  [OK] Embedding 分离充足（std > 0.1）')
    else:
        print(f"  [!] Embedding 分离不足（std={diag['cos_std']:.4f}）")
    print('=' * 72 + '\n')

    config = dict(DEFAULT_EXP1_CONFIG)
    set_global_seed(config['RANDOM_SEED'])
    print(f"\n[CONFIG] {config['EXP1_NAME']}")
    print(f"  PROMPT_VERSION:   {config['EXP1_PROMPT_VERSION']}")
    print(f"  RANDOM_SEED:      {config['RANDOM_SEED']}")
    print(f"  SAMPLE_SIZE:      {config['SAMPLE_SIZE']} 用户")
    print(f"  K_LIST:           {config['K_LIST']}")
    print(f"  MAX_ROUNDS:       {config['MAX_ROUNDS']} 轮")
    print(f"  IMPRESSION_SIZE:  {config['IMPRESSION_SIZE']} 候选集大小")
    print(f"  LAMBDA0:          {config['LAMBDA0']}")
    print(f"  LAMBDA1:          {config['LAMBDA1']}")
    print(f"  GRAPH_K:          {config['GRAPH_K']} (item邻居数)")
    print(f"  MAX_PATH_LENGTH:  {config['MAX_PATH_LENGTH']} (最大路径长度)")
    print(f"  LLM_CANDIDATE_K:  {config['LLM_CANDIDATE_K']} (LLM候选数)")
    print(f"  LLM_PATH_K:       {config['LLM_PATH_K']} (LLM路径长度)")
    print(f"  LLM_TARGET_PRIOR: {config['LLM_TARGET_PRIOR']} (target-aware retrieval)")
    print(f"  LLM_SAFE_MARGIN:  {config['LLM_SAFE_MARGIN']} (LLM接管阈值)")
    print(f"  LLM_RELEASE_RANK: {config['LLM_RELEASE_RANK']} (target提前释放阈值)")
    print(f"  GATING:           {config['ENABLE_GATING']}")
    print(f"  REPLANNING:       {config['ENABLE_REPLANNING']}")
    print(f"  REFLECTION:       {config['ENABLE_REFLECTION']}")
    print(f"  INT_ABLATION:     {config['RUN_INTERNAL_ABLATIONS']}")
    print()
    print('[实验设计]')
    print('  A. Original Backbone   (planner_mode=baseline)')
    print('  B. Main Idea (Path Planning) (planner_mode=greedy_path)')
    print('  C. Ours (LLM + Step3)  (planner_mode=llm_path)')
    if config['RUN_INTERNAL_ABLATIONS']:
        print('  D. Ours w/o Guard       (llm_path, guard disabled)')
        print('  E. Ours w/o Reflection  (llm_path, reflection disabled)')
        print('  F. Ours w/o Rescue      (llm_path, rescue disabled)')
    print('  约束: same input + same candidate + same target')
    print(f"  开关: baseline={config['RUN_BASELINE']} idea={config['RUN_GREEDY_PATH']} llm={config['RUN_LLM_PATH']}")
    print(f"  加速: reduced_logging={config['EXP1_REDUCED_LOGGING']} parallel={config['EXP1_PARALLEL_EXECUTION']}")
    print('  Smoke观察: HR@5/6, IOI@6, acceptance, avg_rounds, variance')
    print()

    run_exp1_pipeline(script_dir, data_loader, news_embeds, embed_type, config)


if __name__ == '__main__':
    main()
