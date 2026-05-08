import os
import sys


PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from qwen_local_feedback import get_feedback_engine  # type: ignore  # noqa: E402,F401