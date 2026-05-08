import hashlib
import os
import random
import subprocess
import sys

import numpy as np


GLOBAL_SEED = 42


def configure_stdout_utf8() -> None:
    try:
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
            import io

            if hasattr(sys.stdout, 'buffer'):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        pass


def stable_seed_from_text(text: str, base_seed: int = GLOBAL_SEED) -> int:
    digest = hashlib.sha256(f'{base_seed}:{text}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little') % (2**32)


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def install_required_packages() -> None:
    required_packages = {
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
        'sentence_transformers': 'sentence-transformers',
        'torch': 'torch',
        'transformers': 'transformers',
        'accelerate': 'accelerate',
        'sentencepiece': 'sentencepiece',
    }

    missing_packages = []
    for module_name, pip_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(pip_name)

    if not missing_packages:
        return

    print(f'[*] 检测到缺失的包: {", ".join(missing_packages)}')
    print('[*] 正在自动安装（适配 Colab 环境）...\n')
    for package in missing_packages:
        print(f'  > pip install {package}')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])
        except subprocess.CalledProcessError:
            print(f'    [!] 安装 {package} 失败，尝试不带 -q 参数重新安装')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
    print('\n[OK] 依赖包安装完成\n')


def print_gpu_info() -> None:
    try:
        import torch

        print(f'[OK] GPU可用: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'     GPU: {torch.cuda.get_device_name(0)}')
            print(f'     内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
    except ImportError:
        print('[!] PyTorch 未安装，将使用 CPU')


def resolve_data_dir(script_dir: str) -> str:
    data_candidates = [
        os.path.join(script_dir, 'MINDsmall_train'),
        os.path.join(script_dir, 'MINDsmall_dev'),
        os.path.join(os.path.dirname(script_dir), 'MINDsmall_train'),
        os.path.join(os.path.dirname(script_dir), 'MINDsmall_dev'),
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'MINDsmall_train'),
        os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'MINDsmall_dev'),
        './MINDsmall_train',
        './MINDsmall_dev',
        '../MINDsmall_train',
        '../MINDsmall_dev',
    ]

    for candidate in data_candidates:
        if os.path.exists(os.path.join(candidate, 'news.tsv')):
            return candidate
    raise FileNotFoundError('未找到 MINDsmall_train 或 MINDsmall_dev 数据集目录')


def validate_data_files(data_dir: str) -> tuple[str, str]:
    news_path = os.path.join(data_dir, 'news.tsv')
    behaviors_path = os.path.join(data_dir, 'behaviors.tsv')
    if not os.path.exists(news_path) or not os.path.exists(behaviors_path):
        raise FileNotFoundError(f'数据文件不完整: {data_dir}')
    return news_path, behaviors_path


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, (np.bool_, np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj