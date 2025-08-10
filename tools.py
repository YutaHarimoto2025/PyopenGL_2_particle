# tools.py

import os, sys
import json
from pathlib import Path
from typing import Any
from omegaconf import OmegaConf
from PyQt6.QtCore import QTimer, pyqtSignal, QCoreApplication
import numpy as np
import git
from datetime import datetime

def create_periodic_timer(parent, slot, interval_ms):
    timer = QTimer(parent)
    timer.timeout.connect(slot)
    timer.start(interval_ms)
    return timer

def make_datetime_file(prefix: str, domain) -> str:
    now = datetime.now()
    fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{prefix}.{domain}"
    filepath  = working_dir / param.save_dir /fname
    filepath.parent.mkdir(parents=True, exist_ok=True) #ディレクトリ作成
    return str(filepath)

# --- NumPy / CuPy自動切替 ---
try:
    import cupy as cp
    if cp.cuda.is_available():
        xp = cp
        USE_CUDA = True
        print("✅ GPU (CuPy) initialized successfully")
    else:
        xp = np
        USE_CUDA = False
        print("⚠️ GPU detected but not available → fallback to NumPy")
except ImportError:
    xp = np
    USE_CUDA = False
    print("⚠️ CuPy not found → using NumPy only")

def _recursive_cast(x, dtype_func):
    """
    任意のリスト・タプル・dict・ndarray・スカラーを
    指定されたdtype_func（例: xp.float32, np.uint32）で再帰変換。
    """
    if hasattr(x, "astype"):
        try:
            return x.astype(dtype_func)
        except Exception:
            return x
    elif isinstance(x, dict):
        return {k: _recursive_cast(v, dtype_func) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        typ = type(x)
        return typ([_recursive_cast(v, dtype_func) for v in x])
    else:
        try:
            return dtype_func(x)
        except Exception:
            return x  # 変換できないものはそのまま

# --- ラッパ関数4つを用意 ---
def xpFloat(x):
    return _recursive_cast(x, xp.float32)

def xpInt(x):
    return _recursive_cast(x, xp.uint32)

def npFloat(x):
    return _recursive_cast(x, np.float32)

def npInt(x):
    return _recursive_cast(x, np.uint32)


# --- 設定ファイル読込 ---
git_repo = git.Repo('.', search_parent_directories=True)
working_dir = Path(git_repo.working_tree_dir)
os.chdir(working_dir) 
sys.path.insert(0, str(working_dir))

PARAM_YAML = working_dir / "param.yaml"
PARAM_JSON = working_dir / "param_changable.json"

if not PARAM_YAML.exists():
    raise FileNotFoundError(f"Configuration file not found: {PARAM_YAML}")
if not PARAM_JSON.exists():
    raise FileNotFoundError(f"Configuration file not found: {PARAM_JSON}")
param = OmegaConf.load(str(PARAM_YAML))

# param_changable: 実行時可変パラメータ用
with open(PARAM_JSON, 'r', encoding='utf-8') as f:
    try:
        param_changable = json.load(f)
    except json.JSONDecodeError:
        param_changable = {}

def update_param_changable():
    """param_changableの内容をparam.jsonで上書き反映（なければ何もしない）"""
    # print("updating param_changable from JSON")
    with open(PARAM_JSON, 'r', encoding='utf-8') as f:
        param_changable.clear()
        param_changable.update(json.load(f))

# --- 乱数生成統一 ---
seed = param.get("seed", 42)
rngxp = xp.random.RandomState(seed)
rngnp = np.random.default_rng(seed)

__all__ = [
    "param", "param_changable", "USE_CUDA", "xp", "np", "rngnp", "rngxp",
    "update_param_changable",
    "xpFloat", "xpInt", "npFloat", "npInt", "working_dir"
]

