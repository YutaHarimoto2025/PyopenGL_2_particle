import glm
import numpy as np

# 2つのクォータニオンを用意（例：単位クォータニオンと90度回転）
q0 = glm.quat(1, 0, 0, 0)  # 単位クォータニオン
angle = np.pi / 2           # 90度回転
axis = glm.vec3(0, 1, 0)    # y軸回り
q1 = glm.angleAxis(angle, axis)  # 90度y軸回転のクォータニオン

t = 0.3  # 補間パラメータ

# --- 1. glm.slerpによる補間 ---
glm_slerp_result = glm.slerp(q0, q1, t)

# --- 2. 式による手計算補間 ---
#! あんまよく分かってない
dot = glm.dot(q0, q1) #内積
# ドット積が負なら補間方向を逆転（GLMも内部でやってる）
if dot < 0.0:
    q1 = -q1
    dot = -dot
# θ
theta = np.arccos(np.clip(dot, -1.0, 1.0))  # クリップで安全
if theta < 1e-6:  # θが小さい場合はlerpで近似
    result_manual = glm.normalize((1-t)*q0 + t*q1)
else:
    s0 = np.sin((1-t)*theta) / np.sin(theta)
    s1 = np.sin(t*theta) / np.sin(theta)
    result_manual = glm.normalize(s0*q0 + s1*q1)

# --- 結果比較 ---
print("GLM slerp:", glm_slerp_result)
print("Manual slerp:", result_manual)

# 各要素の差を確認（ほぼゼロなら一致）
print("Difference:", np.array(glm_slerp_result) - np.array(result_manual))
