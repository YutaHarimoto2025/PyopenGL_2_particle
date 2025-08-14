from collections import deque
from typing import List, Tuple, Dict, Any
import copy
import threading
import time
import glm
import json

from tools import working_dir, param, param_changable, make_datetime_file, xp, np
from create_obj import create_boxes, create_axes, create_balls, one_ball
from object3d import Object3D


class SimBuffer:
    def __init__(self, is_saving_jsonlog: bool):
        # ---- オブジェクトを生成 -----
        self.axes = create_axes()
        self.box = create_boxes(scale=(1, 1, 1))
        self.textural_ball = one_ball(color=(1.0, 1.0, 1.0), texture_path=param_changable["ball_texture"])
        self.balls = create_balls(num=10, radius=0.1)
        self.objects: List[Object3D] = self.box + self.axes + self.balls + self.textural_ball
        # ------------------------------

        # 状態バッファで扱うキーを一元管理
        # rotation は四元数を想定。補間は slerp 特例とする。
        self.state_buffer_keys: tuple[str, str, str] = ("position", "rotation", "scale")

        ball_counter = 1
        self._reindex_objects()
        for obj in self.objects:
            obj.create_gpuBuffer()
            if "ball" in obj.name:  # 球に番号をふる
                idx = obj.name.find("ball") + len("ball")
                obj.name = obj.name[:idx] + str(ball_counter) + obj.name[idx:]
                ball_counter += 1

        # 各時刻の objects の状態（各要素はキー: tuple の dict）
        self.objects_state_buffer: List[List[Dict[str, Any]]] = []

        self.is_saving_jsonlog: bool = is_saving_jsonlog
        self.dt_sim = 0.01  # シミュレーション内部ステップ
        self.t_sim = 0.0
        self.save_time_threshold = 0.0
        self.save_interval_sec = param.jsonl_save_interval_sec  # 保存間隔（秒）
        self.t_multiplier = 1.0
        self.buffer_maxlen = 20

        # cpuスレッド制御用の run ループ継続フラグ
        self._run_flag = threading.Event()  # .set()でON, .clear()でOFF, .is_set()で状態確認, .wait()でONまで待機
        self._onestep_thread = None

        # 保存用 jsonl ファイル
        if self.is_saving_jsonlog:
            self.savefile_path = make_datetime_file(prefix="objectsLOG", domain="jsonl")
            open(self.savefile_path, "w", encoding="utf-8").close()

    # ---------- 状態保存・更新 ----------

    def save_current_state(self) -> None:
        """現在の objects の状態をバッファに記録"""
        state_snapshot = [self._make_state(obj) for obj in self.objects]

        # バッファ上限は外側のステップループで制御しているが、
        # 念のためあふれ対策（古いものから落とす）
        if len(self.objects_state_buffer) >= self.buffer_maxlen:
            self.objects_state_buffer.pop(0)
        self.objects_state_buffer.append(state_snapshot)

    def _onestep(self) -> None:
        """シミュレーションを 1 ステップ進め、状態保存"""
        for obj in self.objects:
            obj.update_posi_rot(self.dt_sim)
        self.save_current_state()

    def _step_loop(self) -> None:
        """バッファが満杯 or フラグが下がるまで onestep を呼び続ける"""
        while self._run_flag.is_set():
            if len(self.objects_state_buffer) < self.buffer_maxlen:
                self._onestep()
            else:
                time.sleep(0.001)  # バッファ満杯時は待機

    def start_stepping(self) -> None:
        """onestep を呼び続けるスレッドを開始"""
        if self._onestep_thread is None or not self._onestep_thread.is_alive():
            self._run_flag.set()  # ON
            self._onestep_thread = threading.Thread(target=self._step_loop)
            self._onestep_thread.start()

    def stop_stepping(self) -> None:
        """ループを止める（フラグを下げる）"""
        self._run_flag.clear()
        if self._onestep_thread is not None:
            self._onestep_thread.join(timeout=1.0)  # 最大1秒待機

    def update_objects(self, dt_frame: float, appended: List[Object3D] | None = None,
                       removed_ids: List[int] | None = None) -> None:
        """
        dt_frame: 今回の描画に必要な「補間秒数」
        → バッファのなかで dt_frame だけ進んだ状態を補間で生成
        """
        self.stop_stepping()  # _onestep ループを止める

        interp_state, buffer_usage_ratio = self.get_interp_state(dt_frame)

        # 整数 id で特定して削除
        if removed_ids:
            for idx in sorted(removed_ids, reverse=True):
                if 0 <= idx < len(self.objects):
                    self.objects.pop(idx)
                    interp_state.pop(idx)

        # オブジェクトの追加
        if appended:
            for obj in appended:
                self.objects.append(obj)
                interp_state.append(self._make_state(obj))

        if removed_ids or appended:
            self._reindex_objects()  # 再インデックス

        # バッファをクリアし、補間済みを先頭に
        self.objects_state_buffer.clear()
        self.objects_state_buffer.append(interp_state)

        # オブジェクトの状態を更新
        self.update_objects_with_state(interp_state)

        # 次のための onestep を開始
        self.start_stepping()

        # JSONL に一行追記
        if self.is_saving_jsonlog and self.t_sim > self.save_time_threshold:
            self.append_jsonl_state(interp_state, buffer_usage_ratio)
            self.save_time_threshold += self.save_interval_sec

    def get_interp_state(self, dt_frame: float) -> Tuple[List[Dict[str, Any]], float]:
        """dt_frame 先の状態を補間して返す（状態リスト, バッファ使用率）"""
        buff_len = len(self.objects_state_buffer)
        if buff_len == 0:
            # バッファが空：現在状態をそのまま返す（異常系のフォールバック）
            print("state buffer is empty. something wrong!")
            return [self._make_state(obj) for obj in self.objects]

        t_sim_forward = dt_frame * self.t_multiplier
        self.t_sim += t_sim_forward

        # 何ステップ分進めるか
        idx0 = int(t_sim_forward / self.dt_sim)  # 切り捨て
        idx1 = idx0 + 1
        alpha = t_sim_forward / self.dt_sim - idx0  # 端数（0~1）

        # バッファ使用率（0~1）。1 を超えるならシミュレーションが追いついていない。
        buffer_usage_ratio = idx1 / buff_len

        if idx1 >= buff_len:
            print("state buffer is not enough. simulation is behind the real time.")
            interp_state = self.objects_state_buffer[-1]
        else:
            state0 = self.objects_state_buffer[idx0]
            state1 = self.objects_state_buffer[idx1]
            interp_state = [self._interpolate_state(s0, s1, alpha) for s0, s1 in zip(state0, state1)]

        return interp_state, buffer_usage_ratio

    def update_objects_with_state(self, state_list: List[Dict[str, Any]]) -> None:
        """interp_state の内容で self.objects を更新"""
        for obj, st in zip(self.objects, state_list):
            # 汎用キー更新（position, rotation, scale）
            for key in self.state_buffer_keys:
                setattr(obj, key, st[key])
            obj.update_model_matrix()  # モデル行列を更新

    def append_jsonl_state(self, state_list: List[Dict[str, Any]], buffer_usage_ratio: float) -> None:
        """
        self.t_sim, 補間状態, obj.name を 1 レコードとして JSONL に追記保存
        ※ thread セーフ（off）な状態で呼ぶこと
        """
        record = {
            "t_sim": self.t_sim,
            "buffer_usage_ratio": buffer_usage_ratio,
            "objects": [
                {
                    "name": obj.name,
                    "StateDict": {key: list(st[key]) for key in self.state_buffer_keys},
                }
                for obj, st in zip(self.objects, state_list)
            ],
        }
        with open(self.savefile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    # ---------- 内部ユーティリティ ----------
    def _reindex_objects(self) -> None:
        for i, obj in enumerate(self.objects):
            obj.obj_id = i

    def _make_state(self, obj: Object3D) -> Dict[str, Any]:
        """Object3D から状態 dict を生成（ディープコピー）"""
        return {
            key: copy.deepcopy(getattr(obj, key))
            for key in self.state_buffer_keys
        }

    @staticmethod
    def _lerp(a, b, alpha: float):
        """線形補間：glm.vec / numpy配列 / スカラに対応（想定）"""
        return a * (1 - alpha) + b * alpha

    def _interpolate_state(self, s0: Dict[str, Any], s1: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """2 状態の補間。rotation だけ slerp、他は線形補間。"""
        out: Dict[str, Any] = {}
        for key in self.state_buffer_keys:
            if key == "rotation":
                # 四元数の球面線形補間
                out[key] = glm.slerp(s0[key], s1[key], alpha)
            else:
                out[key] = self._lerp(s0[key], s1[key], alpha)
        return out
