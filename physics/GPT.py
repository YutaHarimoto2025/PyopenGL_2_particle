# ──────────────────────────────────────────────────────────────────────────────
# File: physics/state.py
# Purpose: 粒子状態(State)と履歴バッファをSoA(Structure of Arrays)で保持
# Python: 3.12 / コメントは日本語 / 可読性重視
# 依存: tools.xp (numpy/cupy切替), tools.np, glm(回転は任意)
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Deque, Dict, Any
from collections import deque

from tools import xp, np  # xpはcupyまたはnumpy、npはnumpy


@dataclass
class ParticleState:
    """粒子群の状態をSoAで保持するコンテナ。
    - pos, vel, acc: (N, 3) の float32 配列
    - mass, radius: (N,) の float32 配列
    - 任意: quat(回転), omega(角速度) は必要になれば拡張
    - history: 直近Kステップのpos/velを保存（レンダ補間や検証用）
    """

    pos: xp.ndarray  # (N,3)
    vel: xp.ndarray  # (N,3)
    acc: xp.ndarray  # (N,3)
    mass: xp.ndarray  # (N,)
    radius: xp.ndarray  # (N,)

    # 物理シミュレーション用設定
    history_len: int = 0

    # 内部: 履歴はCPU側(numpy)に保持（軽量な検証・保存を想定）。
    # 大規模時はhistory_len=0推奨。
    _pos_hist: Deque[np.ndarray] = field(default_factory=deque, init=False)
    _vel_hist: Deque[np.ndarray] = field(default_factory=deque, init=False)

    def __post_init__(self):
        # dtype/shape の最小チェック
        for a in (self.pos, self.vel, self.acc):
            assert a.ndim == 2 and a.shape[1] == 3, "pos/vel/accは(N,3)"
        for a in (self.mass, self.radius):
            assert a.ndim == 1 and a.shape[0] == self.pos.shape[0], "mass/radiusは(N,)"
        
        # float32正規化（GPUメモリ削減）
        f32 = xp.float32
        self.pos = self.pos.astype(f32, copy=False)
        self.vel = self.vel.astype(f32, copy=False)
        self.acc = self.acc.astype(f32, copy=False)
        self.mass = self.mass.astype(f32, copy=False)
        self.radius = self.radius.astype(f32, copy=False)

    @property
    def N(self) -> int:
        return int(self.pos.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """デバッグ用: CPU側の辞書を返す（小規模粒子のみ推奨）。"""
        def as_np(a):
            return a.get() if hasattr(a, "get") else a
        return {
            "pos": as_np(self.pos),
            "vel": as_np(self.vel),
            "acc": as_np(self.acc),
            "mass": as_np(self.mass),
            "radius": as_np(self.radius),
        }

    def record_history(self) -> None:
        """履歴バッファにpos/velを格納（history_len>0のときのみ）。"""
        if self.history_len <= 0:
            return
        pos_cpu = self.pos.get() if hasattr(self.pos, "get") else self.pos
        vel_cpu = self.vel.get() if hasattr(self.vel, "get") else self.vel
        self._pos_hist.append(pos_cpu.copy())
        self._vel_hist.append(vel_cpu.copy())
        if len(self._pos_hist) > self.history_len:
            self._pos_hist.popleft()
            self._vel_hist.popleft()


# ──────────────────────────────────────────────────────────────────────────────
# File: physics/integrators.py
# Purpose: ルンゲクッタ(RK4)とセミインプリシット・オイラー
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Callable, Tuple
from tools import xp

AccelFn = Callable[[xp.ndarray, xp.ndarray, float], xp.ndarray]


class RK4:
    """二階系 x' = v, v' = a(x, v, t) のためのRK4積分器。
    大規模Nでも xやvがベクトル化されていればGPUでそのまま動作。
    """

    def step(self, pos: xp.ndarray, vel: xp.ndarray, accel_fn: AccelFn, dt: float, t: float) -> Tuple[xp.ndarray, xp.ndarray]:
        # k1
        a1 = accel_fn(pos, vel, t)
        k1x = vel
        k1v = a1
        
        # k2
        x2 = pos + 0.5 * dt * k1x
        v2 = vel + 0.5 * dt * k1v
        a2 = accel_fn(x2, v2, t + 0.5 * dt)
        k2x = v2
        k2v = a2

        # k3
        x3 = pos + 0.5 * dt * k2x
        v3 = vel + 0.5 * dt * k2v
        a3 = accel_fn(x3, v3, t + 0.5 * dt)
        k3x = v3
        k3v = a3

        # k4
        x4 = pos + dt * k3x
        v4 = vel + dt * k3v
        a4 = accel_fn(x4, v4, t + dt)
        k4x = v4
        k4v = a4

        pos_next = pos + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        vel_next = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        return pos_next, vel_next


class SemiImplicitEuler:
    """セミインプリシット・オイラー（安定で軽量）"""
    def step(self, pos: xp.ndarray, vel: xp.ndarray, accel_fn: AccelFn, dt: float, t: float) -> Tuple[xp.ndarray, xp.ndarray]:
        acc = accel_fn(pos, vel, t)
        vel_next = vel + dt * acc
        pos_next = pos + dt * vel_next
        return pos_next, vel_next


# ──────────────────────────────────────────────────────────────────────────────
# File: physics/forces.py
# Purpose: 外力モデル（重力、粘性抵抗、位置依存ポテンシャルなど）
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from tools import xp


@dataclass
class ForceConfig:
    gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
    drag_gamma: float = 0.0  # 速度比例の抵抗係数 [1/s]
    max_accel: Optional[float] = None  # 数値安定化用に加速度のクリップ


class ForceModel:
    """位置・速度から加速度場 a(x,v,t) を返すモデル。
    衝突以外の滑らかな力（重力・抵抗・外部ポテンシャルなど）を担当。
    """
    def __init__(self, cfg: ForceConfig):
        self.cfg = cfg
        self._g = xp.asarray(cfg.gravity, dtype=xp.float32)

    def accel(self, pos: xp.ndarray, vel: xp.ndarray, t: float, mass: xp.ndarray) -> xp.ndarray:
        N = pos.shape[0]
        a = xp.zeros_like(pos)
        # 重力: 質量に依らず一定加速度
        a += self._g
        # 粘性抵抗: a += -gamma * v
        if self.cfg.drag_gamma != 0.0:
            a += (-self.cfg.drag_gamma) * vel
        # クリップ（オプション）
        if self.cfg.max_accel is not None:
            mag = xp.linalg.norm(a, axis=1, keepdims=True) + 1e-6
            scale = xp.minimum(1.0, self.cfg.max_accel / mag)
            a = a * scale
        return a


# ──────────────────────────────────────────────────────────────────────────────
# File: physics/collisions.py
# Purpose: 球体の衝突検出・解決（CPU参照実装）。複雑形状用のフックも用意。
# Note: 1e5規模ではGPU実装(セル法のRawKernel等)が必要。まずは正しさ重視のCPU版。
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Iterable, List
import numpy as _np


@dataclass
class CollisionConfig:
    restitution: float = 0.8  # 反発係数(0:非弾性~1:弾性)
    friction: float = 0.0     # 簡易クーロン摩擦（0で無効）
    cell_size: float = 0.2    # 空間分割セルの一辺（半径の~2倍程度）
    max_pairs_per_cell: int = 4096  # 安全装置（過密時の上限）


class SphereColliderCPU:
    """球体同士の衝突をCPU(numpy)で処理。
    - セル分割(Uniform Grid)で近傍候補を限定
    - 各セル+隣接26セルに対して二体判定
    - 速度の衝突応答（位置の軽微補正も実施）

    注意: cupy配列は事前にCPUへ取り出す必要がある（get）。
    大規模化(>1e4)・高密度ではGPU版への置換を推奨。
    """
    def __init__(self, cfg: CollisionConfig):
        self.cfg = cfg

    @staticmethod
    def _cell_index(p: _np.ndarray, h: float) -> _np.ndarray:
        return _np.floor(p / h).astype(_np.int32)

    @staticmethod
    def _neighbors(cx: int, cy: int, cz: int) -> Iterable[Tuple[int, int, int]]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield (cx + dx, cy + dy, cz + dz)

    def resolve(self, pos: _np.ndarray, vel: _np.ndarray, radius: _np.ndarray, mass: _np.ndarray, dt: float) -> None:
        h = self.cfg.cell_size
        cells: Dict[Tuple[int, int, int], List[int]] = {}
        cidx = self._cell_index(pos, h)
        for i, (cx, cy, cz) in enumerate(cidx):
            cells.setdefault((cx, cy, cz), []).append(i)

        R = radius
        M = mass
        e = self.cfg.restitution
        mu = self.cfg.friction

        # 各セル＋近傍セルでペア判定
        for (cx, cy, cz), indices in cells.items():
            for nx, ny, nz in self._neighbors(cx, cy, cz):
                neigh = cells.get((nx, ny, nz))
                if not neigh:
                    continue
                # 同一セルと近傍セルで重複カウントを避けるため、順序付け
                for i_idx in indices:
                    for j_idx in neigh:
                        if j_idx <= i_idx:
                            continue
                        # 球同士の接触判定
                        dp = pos[j_idx] - pos[i_idx]
                        dist2 = float(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2])
                        rsum = float(R[i_idx] + R[j_idx])
                        if dist2 >= rsum * rsum:
                            continue
                        dist = dist2 ** 0.5 if dist2 > 1e-12 else 1e-6
                        n = dp / dist  # 法線

                        # 位置の軽微補正（押し戻し）
                        pen = rsum - dist
                        corr = 0.5 * pen * n  # 等分配
                        pos[i_idx] -= corr
                        pos[j_idx] += corr

                        # 相対速度
                        relv = vel[j_idx] - vel[i_idx]
                        vn = relv.dot(n)
                        if vn > 0.0:
                            continue  # すでに離反

                        # 反発(法線)インパルス
                        invMi = 1.0 / float(M[i_idx])
                        invMj = 1.0 / float(M[j_idx])
                        jn = -(1.0 + e) * vn / (invMi + invMj)
                        impulse_n = jn * n
                        vel[i_idx] -= invMi * impulse_n
                        vel[j_idx] += invMj * impulse_n

                        # 簡易摩擦
                        if mu > 0.0:
                            vt = relv - vn * n
                            vt_norm = _np.linalg.norm(vt) + 1e-12
                            t = vt / vt_norm
                            jt = -mu * jn
                            impulse_t = jt * t
                            vel[i_idx] -= invMi * impulse_t
                            vel[j_idx] += invMj * impulse_t


# 複雑形状のためのフック（頂点群＋BVHなどで拡張予定のスタブ）
class MeshColliderStub:
    def __init__(self):
        pass

    def resolve(self, pos: _np.ndarray, vel: _np.ndarray, vertices: _np.ndarray, faces: _np.ndarray, dt: float) -> None:
        # TODO: AABB階層(BVH)＋GJK/EPA/MTD 等の実装へ拡張
        raise NotImplementedError("Mesh collision is not implemented yet.")


# ──────────────────────────────────────────────────────────────────────────────
# File: physics/simulator.py
# Purpose: 物理エンジン本体。力→積分→衝突→履歴記録の順で更新。
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from typing import Literal, Optional
from tools import xp
from physics.GPT import ParticleState
from physics.integrators import RK4, SemiImplicitEuler
from physics.forces import ForceModel, ForceConfig
from physics.collisions import SphereColliderCPU, CollisionConfig


class PhysicsSimulator:
    """ニュートン力学に基づく粒子系シミュレータ（GPU/CPUハイブリッド）。

    設計方針:
      - 計算の大半（外力・積分）はxp(cupy/numpy)ベクトル化でGPUフレンドリ
      - 衝突はまずCPU参照実装（Uniform Grid）。1e5に伸ばす際はGPU RawKernelへ置換
      - StateはSoAでfloat32。履歴は任意
    """
    def __init__(
        self,
        state: ParticleState,
        dt: float,
        force_cfg: Optional[ForceConfig] = None,
        integrator: Literal["rk4", "semi_euler"] = "rk4",
        collision_cfg: Optional[CollisionConfig] = None,
        collisions_backend: Literal["cpu", "none"] = "cpu",
    ):
        self.state = state
        self.dt = float(dt)
        self.t = 0.0
        self.force_model = ForceModel(force_cfg or ForceConfig())
        self.integrator = RK4() if integrator == "rk4" else SemiImplicitEuler()
        self.collisions_backend = collisions_backend
        self.sphere_collider_cpu = SphereColliderCPU(collision_cfg or CollisionConfig())

    # a(x,v,t) を返す関数を生成（質量は内部で使用可能）
    def _accel_fn(self):
        fm = self.force_model
        mass = self.state.mass
        return lambda x, v, t: fm.accel(x, v, t, mass)

    def step(self, substeps: int = 1) -> None:
        """時間を dt だけ進める。必要に応じてsubsteps回に分割。
        衝突は各サブステップ末尾で解決。
        """
        h = self.dt / max(1, int(substeps))
        acc_fn = self._accel_fn()
        for _ in range(substeps):
            # 積分（GPU/CPUどちらでもxpで動作）
            pos_next, vel_next = self.integrator.step(self.state.pos, self.state.vel, acc_fn, h, self.t)
            self.state.pos = pos_next
            self.state.vel = vel_next

            # 衝突解決（現在はCPU実装）
            if self.collisions_backend == "cpu":
                # cupy→numpyへ一時転送（大規模ではボトルネック: 後日GPU化）
                pos_np = self.state.pos.get() if hasattr(self.state.pos, "get") else self.state.pos
                vel_np = self.state.vel.get() if hasattr(self.state.vel, "get") else self.state.vel
                rad_np = self.state.radius.get() if hasattr(self.state.radius, "get") else self.state.radius
                mass_np = self.state.mass.get() if hasattr(self.state.mass, "get") else self.state.mass

                self.sphere_collider_cpu.resolve(pos_np, vel_np, rad_np, mass_np, h)

                # 返送
                if hasattr(self.state.pos, "set"):
                    self.state.pos.set(pos_np)
                    self.state.vel.set(vel_np)
                else:
                    self.state.pos = pos_np
                    self.state.vel = vel_np

            self.t += h
        
        # 履歴保存（必要なら）
        self.state.record_history()

    # レンダリング用状態（position, rotation, scale）を返す簡易アダプタ
    # rotationは単位四元数、scaleは1ベクトルを返す（必要に応じて拡張）
    def export_render_state(self):
        N = self.state.N
        pos = self.state.pos
        # CPU側へ取り出し
        pos_np = pos.get() if hasattr(pos, "get") else pos
        rot = _unit_quat_array(N)
        scale = _unit_scale_array(N)
        return pos_np, rot, scale


# 補助: 単位四元数と単位スケールを返す（CPU側）
def _unit_quat_array(N: int):
    import numpy as _np
    q = _np.zeros((N, 4), dtype=_np.float32)
    q[:, 3] = 1.0  # w=1, x=y=z=0
    return q


def _unit_scale_array(N: int):
    import numpy as _np
    s = _np.ones((N, 3), dtype=_np.float32)
    return s


# ──────────────────────────────────────────────────────────────────────────────
# File: physics/example_usage.py (任意)
# Purpose: 使い方の最小例（1万粒子の重力落下＋床面反射）
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from tools import xp
    from physics.GPT import ParticleState
    from physics.simulator import PhysicsSimulator
    from physics.forces import ForceConfig
    from physics.collisions import CollisionConfig

    N = 10_000
    rng = xp.random.default_rng(0)

    pos = rng.uniform(low=[-5, 1, -5], high=[5, 10, 5], size=(N, 3)).astype(xp.float32)
    vel = xp.zeros((N, 3), dtype=xp.float32)
    acc = xp.zeros((N, 3), dtype=xp.float32)  # 未使用（計算はforces側）
    mass = xp.full((N,), 1.0, dtype=xp.float32)
    radius = xp.full((N,), 0.05, dtype=xp.float32)

    state = ParticleState(pos, vel, acc, mass, radius, history_len=2)

    sim = PhysicsSimulator(
        state=state,
        dt=1/120,
        force_cfg=ForceConfig(gravity=(0, -9.8, 0), drag_gamma=0.01),
        integrator="rk4",
        collision_cfg=CollisionConfig(restitution=0.6, cell_size=0.12),
        collisions_backend="cpu",
    )

    # 簡易床面(y=0)の処理は外部で行うか、Colliderを拡張して行う
    def floor(pos_np, vel_np):
        import numpy as _np
        hit = pos_np[:, 1] < 0.0
        pos_np[hit, 1] = 0.0
        vel_np[hit, 1] *= -0.5

    # ステップ
    for frame in range(600):
        sim.step(substeps=2)  # 2サブステップで安定化
        # 床面処理（CPU側）
        p = sim.state.pos.get() if hasattr(sim.state.pos, "get") else sim.state.pos
        v = sim.state.vel.get() if hasattr(sim.state.vel, "get") else sim.state.vel
        floor(p, v)
        if hasattr(sim.state.pos, "set"):
            sim.state.pos.set(p)
            sim.state.vel.set(v)

        if frame % 60 == 0:
            print(f"t={sim.t:.3f}, y.mean={p[:,1].mean():.3f}")
