# event_handler.py
from __future__ import annotations

import math
from typing import Optional, Tuple
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QKeyEvent
import glm

class EventHandler:
    """
    GLWidget のインタラクションを集約するクラス。
    - すべての操作を「速度 × dt_frame」でスケーリング（時間基準）
    - キー入力:
        Space         : 一時停止/再開
        矢印（Ctrl無）: 前後左右のパン移動
        PageUp/Down   : カメラ位置 z の上下（target は固定）
        矢印（Ctrl有）: 画面基準パン（従来通り）
    - マウス移動:
        Ctrlなし: 画面端で自動回転（左右=ヨー，上下=仰角）
        Ctrlあり: レイの可視化更新
    - マウス押下（Ctrlあり）:
        左クリック=球生成 / 右クリック=最前面の球削除
    - ホイール: ドリー（ズーム）
    - パラメータは param_changable["event"][...] を参照
    """
    def __init__(self, widget) -> None:
        self.w = widget  # GLWidget への参照

        # 管理対象キー集合
        self.param_keys = {
            "pan_speed", # 単位/秒（前後左右移動）矢印キー
            "yaw_speed", # rad/秒（ヨー）　スクリーン左右端
            "elev_speed", # rad/秒（仰角）スクリーン上下端
            "z_camposi_speed",     # 単位/秒（カメラ位置 z）
            "edge_ratio",  # 画面端の閾値（10%）
            "z_limit_min", # target z の下限
            "z_limit_max"  # target z の上限
        }

        # ハイパラ辞書（実効値）
        self.hyper_pm: dict[str, float] = {}

        # 画面端ジェスチャ用
        self.edge_dir_x = 0  # -1: 左端, 0: なし, 1: 右端
        self.edge_dir_y = 0  # -1: 上端, 0: なし, 1: 下端
        self.edge_timer = QTimer(self.w)
        self.edge_timer.setInterval(16)   # 約60FPS
        self.edge_timer.timeout.connect(self._edge_tick)

        # 初期パラメータ読み込み
        self.refresh_params()

    # ----------------- パラメータ更新 -----------------
    def refresh_params(self) -> None:
        """param_changable から event パラメータを再読込し、hyper_pm と属性を更新。"""
        from tools import param_changable
        for key in self.param_keys:
            self.hyper_pm[key] = float(param_changable["event"][key])
            setattr(self, key, self.hyper_pm[key])

    # ----------------- GLWidget からの委譲イベント -----------------
    def handle_key_press(self, event: QKeyEvent) -> None:
        # PageUp/Down は Ctrl の有無に関係なく優先して処理
        if event.key() in (Qt.Key.Key_PageUp, Qt.Key.Key_PageDown):
            self._move_camera_z(+1 if event.key() == Qt.Key.Key_PageUp else -1)
            self.w.update()
            return

        if event.key() == Qt.Key.Key_Space:
            self._toggle_pause()
            return

        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if press_ctrl:
            self._handle_key_ctrl(event)
        else:
            self._handle_key_noctrl(event)

        self.w.update()
    
    def _stop_edge_rotation(self) -> None:
        """エッジ回転を即停止。外からも呼べるように公開ヘルパ化。"""
        if self.edge_dir_x != 0 or self.edge_dir_y != 0:
            self.edge_dir_x = 0
            self.edge_dir_y = 0
            self.edge_timer.stop()

    def handle_mouse_move(self, event: QMouseEvent) -> None:
        x = float(event.position().x())
        y = float(event.position().y())
        w = float(self.w.width())
        h = float(self.w.height())

        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        if not press_ctrl:
            # スクリーン端 → ヨー/仰角（自動連続）
            left_edge   = x <= self.edge_ratio * w
            right_edge  = x >= (1.0 - self.edge_ratio) * w
            top_edge    = y <= self.edge_ratio * h
            bottom_edge = y >= (1.0 - self.edge_ratio) * h

            new_dir_x = (1 if left_edge else (-1 if right_edge else 0))
            new_dir_y = (-1 if top_edge else (1 if bottom_edge else 0))

            if (new_dir_x != self.edge_dir_x) or (new_dir_y != self.edge_dir_y):
                self.edge_dir_x, self.edge_dir_y = new_dir_x, new_dir_y
                if self.edge_dir_x == 0 and self.edge_dir_y == 0:
                    self.edge_timer.stop()
                else:
                    if not self.edge_timer.isActive():
                        self.edge_timer.start()

            # レイ可視化は Ctrl 時のみ
            if getattr(self.w, "_ray_show", False):
                self.w._ray_show = False
            return

        # Ctrl 押下中: レイの更新・可視化（従来通り）
        self.edge_dir_x = self.edge_dir_y = 0
        self.edge_timer.stop()

        ro, rd = self.w._make_ray(x, y)
        try:
            from graphic_tools import _ray_hit_plane
            P = _ray_hit_plane(ro, rd, plane_point=self.w.plane_point, plane_normal=self.w.plane_normal)
        except Exception:
            P = None

        if P is None:
            if getattr(self.w, "_ray_show", False):
                self.w._ray_show = False
        else:
            self.w._ray_p0, self.w._ray_p1 = P, P + self.w.plane_normal * 0.5
            if not getattr(self.w, "_ray_show", False):
                self.w._ray_show = True

        self.w.update()

    def handle_mouse_press(self, event: QMouseEvent) -> None:
        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if not press_ctrl:
            return

        x, y = float(event.position().x()), float(event.position().y())
        ro, rd = self.w._make_ray(x, y)

        if event.button() == Qt.MouseButton.LeftButton:
            # 球生成
            try:
                from graphic_tools import _ray_hit_plane
                P = _ray_hit_plane(ro, rd, plane_point=self.w.plane_point, plane_normal=self.w.plane_normal)
            except Exception:
                P = None
            if P is None:
                return

            # 範囲チェック
            if abs(P.x - self.w.plane_point.x) > self._checker_len() or abs(P.y - self.w.plane_point.y) > self._checker_len():
                print("hit point is too far from origin, skipping")
                return

            # 生成
            from create_obj import get_oneball_vertices_faces
            from object3d import Object3D
            r = float(self.w.radius)
            center = P + self.w.plane_normal * r
            subdiv = 2 if r < 0.5 else 3
            vertices, tri_indices = get_oneball_vertices_faces(subdiv=subdiv, radius=r)

            nickname = self.w.parent().name_input.toPlainText().strip() if hasattr(self.w.parent(), "name_input") else ""
            if nickname != "":
                name = nickname
            else:
                num_ball = sum(1 for obj in self.w.phys.objects if "ball" in getattr(obj, "name", ""))
                name = f"ball{num_ball+1}"

            if getattr(self.w, "randomize_appended_obj_color", False):
                from tools import rngnp
                color = tuple(rngnp.random(3))
            else:
                c = self.w.parent()._picked_color
                color = (c.redF(), c.greenF(), c.blueF())

            ball = Object3D(
                vertices=vertices,
                tri_indices=tri_indices,
                color=color,
                posi=(center.x, center.y, center.z),
                radius=r,
                is_move=True,
                name=name,
            )
            self.w.makeCurrent()
            ball.create_gpuBuffer()
            self.w.appended_object.append(ball)

        elif event.button() == Qt.MouseButton.RightButton:
            # 球削除（最前面）
            from graphic_tools import _ray_hit_sphere
            import glm as _glm
            hit_idx = -1
            hit_t = float("inf")
            removed_obj = None
            for obj in self.w.phys.objects:
                if "ball" not in getattr(obj, "name", ""):
                    continue
                center = _glm.vec3(*obj.position)
                radius = obj.radius if getattr(obj, "radius", None) is not None else max(obj.scale.x, obj.scale.y, obj.scale.z)
                t = _ray_hit_sphere(ro, rd, center, radius)
                if t is not None and t < hit_t:
                    hit_t = t
                    hit_idx = obj.obj_id
                    removed_obj = obj
            if hit_idx >= 0:
                self.w.makeCurrent()
                removed_obj.destroy_gpuBuffer()
                self.w.removed_object_idx.append(hit_idx)

        self.w.update()

    def handle_wheel(self, event: QMouseEvent) -> None:
        # ドリー（ズーム）。ここは OS 依存差を考慮し、角度 delta を直接利用
        angle = event.angleDelta().y()
        length = glm.length(self.w.cam_target - self.w.cam_posi)
        direc = glm.normalize(self.w.cam_target - self.w.cam_posi)
        step = 0.1 if angle > 0 else -0.1
        length = max(0.1, min(length - step, 100.0))
        self.w.cam_posi = self.w.cam_target - direc * length
        self.w.update()

    # ----------------- 内部: キー処理 -----------------
    def _handle_key_noctrl(self, event: QKeyEvent) -> None:
        # 矢印（Ctrlなし）＝ 前後左右のパン（時間基準）
        forward, left = self._forward_left_xy()
        dt = self._dt()
        s = self.pan_speed * dt
        if event.key() == Qt.Key.Key_Up:
            self._pan(forward * s)
        elif event.key() == Qt.Key.Key_Down:
            self._pan(-forward * s)
        elif event.key() == Qt.Key.Key_Left:
            self._pan(left * s)
        elif event.key() == Qt.Key.Key_Right:
            self._pan(-left * s)

    def _handle_key_ctrl(self, event: QKeyEvent) -> None:
        #! 今後他の機能にする（現状は画面基準パンのまま）
        forward, left = self._forward_left_xy()
        dt = self._dt()
        s = self.pan_speed * dt
        if event.key() == Qt.Key.Key_Up:
            self._pan(forward * s)
        elif event.key() == Qt.Key.Key_Down:
            self._pan(-forward * s)
        elif event.key() == Qt.Key.Key_Left:
            self._pan(left * s)
        elif event.key() == Qt.Key.Key_Right:
            self._pan(-left * s)

    # ----------------- 内部: 共通ユーティリティ -----------------
    def _toggle_pause(self) -> None:
        self.w.is_paused = not self.w.is_paused
        if self.w.is_paused:
            self.w.pause_start_time = self._perf_counter()
        else:
            if self.w.pause_start_time is not None:
                paused_time = self._perf_counter() - self.w.pause_start_time
                self.w.paused_duration += paused_time
                self.w.pause_start_time = None

    def _perf_counter(self) -> float:
        import time
        return time.perf_counter()

    def _dt(self) -> float:
        # GLWidget 側で毎フレーム更新される dt_frame [sec]
        try:
            return float(getattr(self.w, "dt_frame", 1.0 / 60.0))
        except Exception:
            return 1.0 / 60.0

    def _horizontal_dist(self) -> float:
        dx = float(self.w.cam_target.x - self.w.cam_posi.x)
        dy = float(self.w.cam_target.y - self.w.cam_posi.y)
        return math.hypot(dx, dy)

    def _change_elevation(self, delta_angle: float) -> None:
        # 仰角を delta_angle だけ変更し、target.z を再計算
        horiz = self._horizontal_dist()
        dz = float(self.w.cam_target.z - self.w.cam_posi.z)
        elev = math.atan2(dz, horiz) + delta_angle
        new_z = float(self.w.cam_posi.z) + math.tan(elev) * horiz
        self.w.cam_target.z = max(self.z_limit_min, min(new_z, self.z_limit_max))

    def _yaw_rotate(self, angle: float) -> None:
        # カメラ位置を中心に、注視ベクトルを Z 軸回りに回転
        direction = self.w.cam_target - self.w.cam_posi
        rot = glm.rotate(glm.mat4(1), float(angle), glm.vec3(0, 0, 1))
        dir_rot = glm.vec3(rot * glm.vec4(direction, 1.0))
        self.w.cam_target = self.w.cam_posi + dir_rot

    def _forward_left_xy(self) -> Tuple[glm.vec3, glm.vec3]:
        """画面前(forward)・左(left) の XY 平面単位ベクトル。"""
        direction = self.w.cam_target - self.w.cam_posi
        xy = glm.vec3(direction.x, direction.y, 0.0)
        if glm.length(xy) == 0:
            xy = glm.vec3(0, 1, 0)
        forward = glm.normalize(xy)
        left = glm.vec3(-forward.y, forward.x, 0.0)
        return forward, left

    def _pan(self, move_vec: glm.vec3) -> None:
        """pos と target を同量移動（パン）"""
        self.w.cam_posi += move_vec
        self.w.cam_target += move_vec

    def _move_camera_z(self, sign: int) -> None:
        """カメラ位置 z のみ移動。target は固定。将来仰角で上下させたっていい"""
        dt = self._dt()
        dz = float(sign) * self.z_camposi_speed * dt
        new_z = float(self.w.cam_posi.z) + dz
        # z制限は任意（必要なら適用）
        self.w.cam_posi.z = new_z

    def _checker_len(self) -> float:
        try:
            from tools import param_changable
            return float(param_changable["checkerboard"]["length"])
        except Exception:
            return 10.0

    # ----------------- 画面端ジェスチャ（ヨー/仰角） -----------------
    def _edge_tick(self) -> None:
        if self.edge_dir_x == 0 and self.edge_dir_y == 0:
            return
        dt = self._dt()
        # 左右端 → ヨー回転
        if self.edge_dir_x != 0:
            self._yaw_rotate(self.edge_dir_x * self.yaw_speed * dt)
        # 上下端 → 仰角変更
        if self.edge_dir_y != 0:
            # 画面上端(-1)で仰角を上げるので -sign
            self._change_elevation(-self.edge_dir_y * self.elev_speed * dt)
        self.w.update()
