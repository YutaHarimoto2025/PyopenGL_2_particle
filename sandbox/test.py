import numpy as np
import cupy as xp

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

A =  [(0, 0, 0), (1, 0, 0)]
print(type(A))
print(type(npFloat(A)))
print(type(np.asarray(npFloat(A))[0][0]))
print(type(np.asarray(A)))