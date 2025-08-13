# event_handler.py
from __future__ import annotations

import math
from typing import Optional
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QKeyEvent
import glm

class EventHandler:
    """
    GLWidget のインタラクションを責務分離して扱うハンドラクラス。
    - キー入力: 一時停止、仰角変更、Z軸回転（ヨー）、パン（Ctrl+矢印）
    - マウス移動: エッジパン（左右20%ゾーン）、Ctrl+移動でレイの可視化
    - マウス押下: Ctrl+左で球生成、Ctrl+右で球削除
    - ホイール: ドリー（ズーム）
    """
    def __init__(self, widget) -> None:
        self.w = widget  # GLWidget への参照

        # 感度・制限（必要に応じて param.yaml へ移してもよい）
        self.angle_step = 0.03           # ラジアン。上下キーの仰角変化量
        self.yaw_step = 0.03             # ラジアン。左右キーのヨー角変化量（Z軸回転）
        self.pan_step = 0.1              # Ctrl+矢印やエッジパンの1ステップ移動量
        self.z_limit_min = -10.0         # 注視点Zの下限（仰角の負側を許可）
        self.z_limit_max =  10.0         # 注視点Zの上限

        # エッジパン設定（右/左20%で自動連続パン）
        self.edge_ratio = 0.2            # 画面幅の20%をエッジとする
        self.edge_pan_dir = 0            # -1: 左へ、0: 停止、1: 右へ
        self.edge_pan_timer = QTimer(self.w)
        self.edge_pan_timer.setInterval(16)  # 約60FPS
        self.edge_pan_timer.timeout.connect(self._edge_pan_tick)

    # ----------------- 公開: GLWidget から委譲されるイベント -----------------
    def handle_key_press(self, event: QKeyEvent) -> None:
        # スペース: 一時停止/再開（GLWidgetの状態を直接更新）
        if event.key() == Qt.Key.Key_Space:
            self._toggle_pause()
            return

        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if press_ctrl:
            self._handle_key_ctrl(event)
        else:
            self._handle_key_noctrl(event)

        self.w.update()

    def handle_mouse_move(self, event: QMouseEvent) -> None:
        x = float(event.position().x())
        w = float(self.w.width())

        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)

        # Ctrl が押されていない場合はエッジパン制御（左右20%）
        if not press_ctrl:
            if x >= (1.0 - self.edge_ratio) * w:
                self._set_edge_pan_dir(1)   # 右へ（Ctrl+→ と同じ）
            elif x <= self.edge_ratio * w:
                self._set_edge_pan_dir(-1)  # 左へ（Ctrl+← と同じ）
            else:
                self._set_edge_pan_dir(0)   # 中央域は停止

            # レイ表示はCtrl時のみ
            if self.w._ray_show:
                self.w._ray_show = False
            return

        # Ctrl 押下中: レイの更新・可視化
        self._set_edge_pan_dir(0)  # エッジパンは停止
        ro, rd = self.w._make_ray(x, float(event.position().y()))
        P = self.w._ray_hit_plane(ro, rd) if hasattr(self.w, "_ray_hit_plane") else None

        # 既存のレイ可視化ロジック（GLWidget版に合わせて処理）
        from graphic_tools import _ray_hit_plane
        P = _ray_hit_plane(ro, rd, plane_point=self.w.plane_point, plane_normal=self.w.plane_normal)
        if P is None:
            if self.w._ray_show:
                self.w._ray_show = False
        else:
            self.w._ray_p0, self.w._ray_p1 = P, P + self.w.plane_normal * 0.5
            if not self.w._ray_show:
                self.w._ray_show = True

        self.w.update()

    def handle_mouse_press(self, event: QMouseEvent) -> None:
        press_ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if not press_ctrl:
            return

        # 以降は Ctrl 押下時のみ（既存GLWidgetのロジックを移植）
        x, y = float(event.position().x()), float(event.position().y())
        ro, rd = self.w._make_ray(x, y)

        if event.button() == Qt.MouseButton.LeftButton:
            # 生成
            from graphic_tools import _ray_hit_plane
            P = _ray_hit_plane(ro, rd, plane_point=self.w.plane_point, plane_normal=self.w.plane_normal)
            if P is None:
                return

            # 平面外チェック（checkerboard長で制限）
            L = float(self.w.parent().param_changable["checkerboard"]["length"]) if hasattr(self.w.parent(), "param_changable") else None
            if abs(P.x - self.w.plane_point.x) > self._checker_len() or abs(P.y - self.w.plane_point.y) > self._checker_len():
                print("hit point is too far from origin, skipping")
                return

            # 球生成（GLWidgetの実装をそのまま利用）
            from create_obj import get_oneball_vertices_faces
            from object3d import Object3D
            r = float(self.w.radius)
            center = P + self.w.plane_normal * r
            subdiv = 2 if r < 0.5 else 3
            vertices, tri_indices = get_oneball_vertices_faces(subdiv=subdiv, radius=r)

            nickname = self.w.parent().name_input.toPlainText().strip()
            if nickname != "":
                name = nickname
            else:
                num_ball = sum(1 for obj in self.w.phys.objects if "ball" in obj.name)
                name = f"ball{num_ball+1}"

            if self.w.randomize_appended_obj_color:
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
            # 削除（レイと最初に当たる球）
            from graphic_tools import _ray_hit_sphere
            hit_idx = -1
            hit_t = float("inf")
            removed_obj = None
            for obj in self.w.phys.objects:
                if "ball" not in getattr(obj, "name", ""):
                    continue
                center = glm.vec3(*obj.position)
                radius = obj.radius if getattr(obj, "radius", None) is not None else max(obj.scale.x, obj.scale.y, obj.scale.z) * 1.0
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
        angle = event.angleDelta().y()
        length = glm.length(self.w.cam_target - self.w.cam_posi)
        direc = glm.normalize(self.w.cam_target - self.w.cam_posi)
        sensitivity = 0.1
        delta = -sensitivity if angle < 0 else sensitivity
        length = max(0.1, min(length - delta, 100.0))
        self.w.cam_posi = self.w.cam_target - direc * length
        self.w.update()

    # ----------------- 内部: キー処理の分解 -----------------
    def _handle_key_noctrl(self, event: QKeyEvent) -> None:
        # 注視ベクトルなどの準備
        direction = self.w.cam_target - self.w.cam_posi

        if event.key() == Qt.Key.Key_Up:
            self._change_elevation(+self.angle_step)
        elif event.key() == Qt.Key.Key_Down:
            self._change_elevation(-self.angle_step)
        elif event.key() == Qt.Key.Key_Left:
            self._yaw_rotate(direction, +self.yaw_step)   # Z軸まわり反時計回り
        elif event.key() == Qt.Key.Key_Right:
            self._yaw_rotate(direction, -self.yaw_step)   # Z軸まわり時計回り

    def _handle_key_ctrl(self, event: QKeyEvent) -> None:
        # 画面基準パン（Ctrl+矢印）
        forward, left = self._forward_left_xy()
        step_f = self.pan_step
        step_l = self.pan_step
        if event.key() == Qt.Key.Key_Up:
            self._pan(forward * step_f)
        elif event.key() == Qt.Key.Key_Down:
            self._pan(-forward * step_f)
        elif event.key() == Qt.Key.Key_Left:
            self._pan(left * step_l)
        elif event.key() == Qt.Key.Key_Right:
            self._pan(-left * step_l)

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

    def _horizontal_dist(self) -> float:
        dx = float(self.w.cam_target.x - self.w.cam_posi.x)
        dy = float(self.w.cam_target.y - self.w.cam_posi.y)
        return math.hypot(dx, dy)

    def _change_elevation(self, delta_angle: float) -> None:
        # 現在の仰角から角度一定で更新 → zを再計算
        horiz = self._horizontal_dist()
        dz = float(self.w.cam_target.z - self.w.cam_posi.z)
        elev = math.atan2(dz, horiz)
        elev += delta_angle
        new_z = float(self.w.cam_posi.z) + math.tan(elev) * horiz
        # z制限（正負両方許可）
        new_z = max(self.z_limit_min, min(new_z, self.z_limit_max))
        self.w.cam_target.z = new_z

    def _yaw_rotate(self, direction: glm.vec3, angle: float) -> None:
        rot = glm.rotate(glm.mat4(1), angle, glm.vec3(0, 0, 1))
        dir_rotated = glm.vec3(rot * glm.vec4(direction, 1.0))
        self.w.cam_target = self.w.cam_posi + dir_rotated

    def _forward_left_xy(self) -> tuple[glm.vec3, glm.vec3]:
        """画面前方向(forward)と左方向(left)の単位ベクトル（XY平面）。"""
        direction = self.w.cam_target - self.w.cam_posi
        xy = glm.vec3(direction.x, direction.y, 0.0)
        if glm.length(xy) == 0:
            xy = glm.vec3(0, 1, 0)
        forward = glm.normalize(xy)
        left = glm.vec3(-forward.y, forward.x, 0.0)  # XY平面で90度回転
        return forward, left

    def _pan(self, move_vec: glm.vec3) -> None:
        """pos と target を同量だけ平行移動（パン）。"""
        self.w.cam_posi += move_vec
        self.w.cam_target += move_vec

    def _checker_len(self) -> float:
        # param_changable を直接参照できない場合は安全側の固定値を使う
        try:
            from tools import param_changable
            return float(param_changable["checkerboard"]["length"])
        except Exception:
            return 10.0

    # ----------------- 内部: エッジパン -----------------
    def _set_edge_pan_dir(self, d: int) -> None:
        if d == self.edge_pan_dir:
            return
        self.edge_pan_dir = d
        if d == 0:
            self.edge_pan_timer.stop()
        else:
            if not self.edge_pan_timer.isActive():
                self.edge_pan_timer.start()

    def _edge_pan_tick(self) -> None:
        if self.edge_pan_dir == 0:
            return
        # Ctrl+←/→ と同じ処理：左は +left、右は -left
        _, left = self._forward_left_xy()
        step = self.pan_step
        move = left * step if self.edge_pan_dir < 0 else -left * step
        self._pan(move)
        self.w.update()
