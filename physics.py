from collections import deque
from typing import List
import copy
import threading
import time
import glm
import json

from tools import working_dir, param, param_changable, make_datetime_file, xp, np
from create_obj import create_boxes, create_axes, create_balls
from object3d import Object3D

class Physics:
    def __init__(self, is_saving):
        self.is_saving: bool = is_saving
        self.axes = create_axes()
        self.box = create_boxes(scale=(1, 1, 1))
        self.balls = create_balls(num=10, radius=0.1)
        self.objects: List[Object3D] = self.box + self.axes + self.balls
        self.objects_state_buffer: List[List[dict]] = []
        self.dt_sim = 0.001
        self.t_sim = 0.0
        self.t_multiplier = 1.0
        self.buffer_maxlen = 100
        
        self._run_flag = threading.Event() #cpuスレッド制御用のrunループ継続フラグ
        # .set()でON, .clear()でOFF, .is_set()で状態確認, .wait()でONまで待機
        self._onestep_thread = None
        
        #保存用jsonlファイル
        if self.is_saving:
            self.savefile_path = make_datetime_file(prefix="objectsLOG", domain="jsonl")
            open(self.savefile_path, "w", encoding="utf-8").close()

    def save_current_state(self):
        """ 現在のobjectsの状態をバッファに記録 """
        state_snapshot = [
            {
                "position": copy.deepcopy(obj.position),
                "rotation": copy.deepcopy(obj.rotation),
                "scale":    copy.deepcopy(obj.scale),
            }
            for obj in self.objects
        ]
        self.objects_state_buffer.append(state_snapshot)

    def _onestep(self):
        """シミュレーション1ステップ進め、状態保存"""
        for obj in self.objects:
            obj.update_posi_rot(self.dt_sim)
        self.save_current_state()
        
    def _step_loop(self):
        """ バッファが満杯になる or フラグが下がるまでonestepを呼び続ける """
        while self._run_flag.is_set():
            if len(self.objects_state_buffer) < self.buffer_maxlen:
                self._onestep()
            else:
                time.sleep(0.001)  # バッファ満杯時は待機
        
    def start_stepping(self):
        """ onestepをずっと呼び続けるスレッドを開始 """
        if self._onestep_thread is None or not self._onestep_thread.is_alive():
            self._run_flag.set() #ON
            self._onestep_thread = threading.Thread(target=self._step_loop)
            self._onestep_thread.start()
            
    def stop_stepping(self):
        """ ループを止める（フラグを下げる） """
        self._run_flag.clear()
        if self._onestep_thread is not None:
            self._onestep_thread.join(timeout=1.0) #最大1秒待機して終了を待つ

    def update_objects(self, t: float, dt_frame: float):
        """
        t: 現実時間またはシミュレーション時間
        dt_frame: 今回の描画に必要な「補間秒数」
        → bufferのなかで dt_frame だけ進んだ状態を線形補間で生成
        """
        self.stop_stepping() #_onestepループを止める
        
        interp_state, buffer_usage_ratio = self.get_interp_state(dt_frame)

        # バッファをクリアし補間済みを先頭に
        self.objects_state_buffer.clear()
        self.objects_state_buffer.append(interp_state)
        
        # オブジェクトの状態を更新
        self.update_objects_with_state(interp_state)
        
        # 次のための_onestepを開始
        self.start_stepping()   
        
        # JSONLに一行追記
        if self.is_saving:
            self.append_jsonl_state(interp_state, buffer_usage_ratio)
        
    def get_interp_state(self, dt_frame: float) -> List[dict]:
        buff_len = len(self.objects_state_buffer)
        if buff_len == 0: # バッファが空、現在状態をそのまま返す
            print("state buffer is empty. something wrong!")
            return [
                {
                    "position": copy.deepcopy(obj.position),
                    "rotation": copy.deepcopy(obj.rotation),
                    "scale":    copy.deepcopy(obj.scale),
                }
                for obj in self.objects
            ]
        t_sim_foreard = dt_frame * self.t_multiplier
        self.t_sim += t_sim_foreard
        
        idx0 = int(t_sim_foreard / self.dt_sim) #切り捨て
        idx1 = idx0 + 1
        
        alpha = t_sim_foreard / self.dt_sim - idx0 #端数（0~1）
        
        #バッファ使用率（0~1）1を超過するならシミュレーションが追いついてない
        buffer_usage_ratio = idx1 / buff_len 
        if idx1 >= buff_len:
            print("state buffer is not enough. simulation is behind the real time.")
            interp_state =  self.objects_state_buffer[-1]
        else:
            # state~hogeはオブジェクト数に等しい長さのリストで、各要素は辞書
            state0 = self.objects_state_buffer[idx0]
            state1 = self.objects_state_buffer[idx1]
            interp_state = []
            for s0, s1 in zip(state0, state1):
                pos = s0["position"] * (1 - alpha) + s1["position"] * alpha
                rot = glm.slerp(s0["rotation"], s1["rotation"], alpha)
                scale = s0["scale"] * (1 - alpha) + s1["scale"] * alpha
                interp_state.append({
                    "position": pos,
                    "rotation": rot,
                    "scale":    scale,
                })
        return interp_state, buffer_usage_ratio
        
    def update_objects_with_state(self, state: List[dict]) -> None:
        """
        interp_stateの内容でself.objectsを更新
        """
        for obj, state in zip(self.objects, state):
            obj.position = state['position']
            obj.rotation = state['rotation']
            obj.scale = state['scale']
            obj.update_model_matrix()  # モデル行列を更新
            
    def append_jsonl_state(self, state: List[dict], buffer_usage_ratio: float) -> None: 
        """
        self.t_sim, interp_state, obj.name を1レコードとしてJSONLに追記保存する
        必ずthreadセーフ（off）な状態で呼ぶこと
        """
        # 1ステップ分をまとめて保存（リストで各粒子・オブジェクト分を持つ）
        record = {
            "t_sim": self.t_sim,
            "buffer_usage_ratio": buffer_usage_ratio,
            "objects": [
                {
                    "name": obj.name,
                    "StateDict": {
                        "position": list(s["position"]),
                        "rotation": list(s["rotation"]),
                        "scale": list(s["scale"]),
                    }
                }
                for obj, s in zip(self.objects, state)
            ]
        }
        with open(self.savefile_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")