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
    fname = f"{now.strftime('%Y%m%d_%H%M%S')}_{param.case_name}_{prefix}.{domain}"
    filepath  = working_dir / param.save_dir /fname
    filepath.parent.mkdir(parents=True, exist_ok=True) #ディレクトリ作成
    return str(filepath)

# --- NumPy / CuPy自動切替 ---
try:
    import cupy as cp
    if cp.cuda.is_available():
        cp.random.RandomState(0)  # GPU使用可能かテスト
        cp.random.default_rng
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
    

# --- データ型変換の再帰関数 ---
def _recursive_cast(x, dtype):
    """
    任意のデータ構造を指定された dtype に再帰的に変換する。
    """
    # ndarray/cp.ndarray の場合
    if hasattr(x, "astype"):
        try:
            return x.astype(dtype)
        except Exception:
            return x
    # 辞書の場合
    elif isinstance(x, dict):
        return {k: _recursive_cast(v, dtype) for k, v in x.items()}
    # リスト・タプルの場合
    elif isinstance(x, (list, tuple)):
        typ = type(x)
        return typ([_recursive_cast(v, dtype) for v in x])
    # スカラー値などの場合
    else:
        try:
            # dtype 自体がコンストラクタとして機能する（np.float32(1.0) など）
            return dtype(x)
        except Exception:
            return x

def to_float(x, use_xp=True, precision=32):
    """
    float型への変換。backend(xp/np)と精度(32/64)を選択可能。
    """
    lib = xp if use_xp else np
    dtype = lib.float32 if precision == 32 else lib.float64
    return _recursive_cast(x, dtype)

def to_int(x, use_xp=True, precision=32, unsigned=True):
    """
    int型への変換。符号あり/なし、backend(xp/np)、精度(32/64)を選択可能。
    """
    lib = xp if use_xp else np
    if unsigned:
        dtype = lib.uint32 if precision == 32 else lib.uint64
    else:
        dtype = lib.int32 if precision == 32 else lib.int64
    return _recursive_cast(x, dtype)



# --- 設定ファイル読込 ---
git_repo = git.Repo(Path(__file__).resolve(), search_parent_directories=True)
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
rngnp = np.random.default_rng(seed)
rngxp = xp.random.default_rng(seed)

__all__ = [
    "param", "param_changable", "USE_CUDA", "xp", "np", "rngnp", "rngxp",
    "update_param_changable",
    "to_float", "to_int", "working_dir"
]