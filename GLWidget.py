# GLWidget.py

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QPen, QColor, QPainterPath, QMouseEvent, QKeyEvent
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from OpenGL import GL
from OpenGL.GLU import gluUnProject
import glm
from typing import List, Optional
from pathlib import Path
import time

from tools import xp, np, create_periodic_timer, param, param_changable, working_dir, rngnp  # CuPy/NumPy, 各種ユーティリティ, ハイパーパラメータ
from graphic_tools import load_shader, compute_normals, _ray_hit_plane, _ray_hit_sphere  # シェーダ読み込み、法線計算、レイキャスト
from create_obj import create_boxes, create_axes, get_oneball_vertices_faces  # オブジェクト生成はここに分離
from object3d import Object3D  # 3Dオブジェクト定義
from movie_ffepeg import MovieFFmpeg
from simulation_buffer import SimBuffer  # 物理シミュレーションデータ
from rendering import apply_common_rendering_settings, ObjectRenderer, create_nonobject_renderers

class GLWidget(QOpenGLWidget):
    """
    OpenGLによる3D描画ウィジェット。
    CuPy/NumPy自動切替（xp）で物理データを持ち、GL描画時にのみNumPyへ変換。
    Object3D定義・生成はcreate_obj.pyへ分離。
    ハイパーパラメータはparam.yaml/tools経由で一元管理。
    """
    fpsUpdated = pyqtSignal(int) #クラス変数として定義

    def __init__(self, status_callback, parent=None) -> None:
        super().__init__(parent)
        self._status_callback = status_callback
        self.total_frame: int = 0                # 保存する総フレーム数
        self.aspect: float = 1.0                 # ウィンドウアスペクト比
        self.show_labels: bool = False  # ラベル表示フラグ
        self.randomize_appended_obj_color: bool = False  # ランダム色フラグ
        self.radius: float = 0.1  # 半径（スライダーで調整）
        
        self.previous_frameCount:int = 0
        
        self.appended_object = []
        self.removed_object_idx = []
        self._ray_show = False
        self._ray_p0 = glm.vec3(0, 0, 0)  # レイの始点終点仮設定
        self._ray_p1 = glm.vec3(0, 0, 0)
        
        #　球生成，レイキャスト用の平面
        self.plane_point  = glm.vec3(0,0,0)
        self.plane_normal = glm.vec3(0,0,1)

        self.cam_posi = glm.vec3(2, -2, 2)  # カメラ位置のデフォ値
        self.cam_target = glm.vec3(0, 0, 0)  # 注視点
        
    def initializeGL(self) -> None:
        """
        OpenGL初期化処理。シェーダコンパイル、オブジェクト生成、背景色設定、動画保存準備など。
        """
        # --- シェーダプログラム読み込み・コンパイル ---
        self.renderer = ObjectRenderer()
        self.checker, self.ray, self.cam_target_point = create_nonobject_renderers(target_position=self.cam_target)
        
        self.setMouseTracking(True) #クリックしなくてもマウス移動イベントを受け取れる
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # キーボードイベントを受け取るためにフォーカスを強制的に設定
        # --- 動画保存用ffmpeg準備 ---
        self.is_saving = bool(param.is_saving)
        self.frameCount = 0
        if self.is_saving:
            self.ffmpeg = MovieFFmpeg(self.width(), self.height())
        
        self.simbuff = SimBuffer(self.is_saving)  # 物理シミュレーションデータ
        self.simbuff.start_stepping()  # シミュレーションスレッド開始
        self._status_callback() # 初期状態のステータスを表示
        
        self.start_time = time.perf_counter()  # 描画開始時刻
        self.previous_time = self.start_time
        self.is_paused: bool = False  # 一時停止フラグ
        self.total_paused_time = 0.0
        self.pause_start_time = None

        self.record_fps_timer = create_periodic_timer(self, self.FpsTimer, 1000)
        self.ctrl_fps_timer = create_periodic_timer(self, self.update, max(5, 1000//int(param_changable["fps"])))

    def resizeGL(self, w: int, h: int) -> None:
        """
        ウィンドウサイズ変更時に自動で呼ばれる。OpenGLビューポートも更新。
        """
        GL.glViewport(0, 0, w, h)
        self.aspect = w / h if h > 0 else 1

    def paintGL(self) -> None:
        """
        3Dシーンの描画、QPainterによるラベル描画、動画保存処理。
        """
        GL.glClearColor(*param_changable["bg_color"]) #背景色
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST) #これを毎回呼ばないと追加objが遠くても全面に出てしまう

        # 透視投影行列
        self.view = glm.lookAt(self.cam_posi, self.cam_target, glm.vec3(0,0,1)) #カメラ位置，注視点， 上方向
        self.proj = glm.perspective(glm.radians(param_changable["fov"]), self.aspect, 0.01, 100.0) #視野角， アスペクト比，近接面，遠方面
        
        # 正射影
        # cam_posi = glm.vec3(0, 0, 10)
        # self.view = glm.lookAt(cam_posi, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # self.proj = glm.ortho(-5.0, 5.0, -5.0, 5.0, 0.01, 100.0)
        
        self.renderer.set_common(self.cam_posi, self.view, self.proj)
        self.checker.draw(self.view, self.proj, 
                additional_uniform_dict={"L": float(param_changable["checkerboard"]["length"])}) 
        self.cam_target_point.draw(self.view, self.proj, 
                additional_uniform_dict={"position": self.cam_target})
        if self._ray_show:
            self.ray.draw(self.view, self.proj, 
                additional_uniform_dict={"uP0":self._ray_p0, "uP1":self._ray_p1})
        
        current_time = time.perf_counter()
        # t = current_time - self.start_time - self.total_paused_time # 経過時間 [秒]
        dt_frame = current_time - self.previous_time  # 前フレームからの経過時間 [秒]

        if not self.is_paused:
            # --- シミュレーション更新 ---
            self.simbuff.update_objects(dt_frame, appended=self.appended_object, removed_ids=self.removed_object_idx)
            if self.appended_object:
                self._status_callback(text = "オブジェクト追加しました")
                self.appended_object.clear()
            if self.removed_object_idx:
                self._status_callback(text = "オブジェクト削除しました")
                self.removed_object_idx.clear()

        # --- オブジェクトの描画 ---
        for obj in self.simbuff.objects:
            self.renderer.set_each(obj)   # uModel / uNormalMatrix / uColor             # uColor
            if obj.name == "box":
                GL.glDisable(GL.GL_CULL_FACE)
                self.renderer.draw(obj) #両面描く
                GL.glEnable(GL.GL_CULL_FACE)
            else:
                self.renderer.draw(obj)

        # --- QPainterでラベル描画 ---
        if self.show_labels:
            painter = QPainter(self)
            font = QFont("Noto Sans CJK JP", 20, QFont.Weight.Normal)
            painter.setFont(font)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            for obj in self.simbuff.objects:
                pos = obj.localframe_to_window(self.view, self.proj, (self.width(), self.height()))
                r, g, b, a = [int(c*255) for c in obj.color]
                 # 輪郭（黒線）
                painter.setPen(QPen(Qt.GlobalColor.black))
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    painter.drawText(pos[0] + dx, pos[1] + dy, obj.name)

                # 中身（指定色）
                painter.setPen(QColor(r, g, b))
                painter.drawText(pos[0], pos[1], obj.name)
            painter.end()

        # --- フレームカウント管理・動画保存処理 ---
        if self.is_saving:
            self.ffmpeg.step(self.frameCount)
        
        self.frameCount += 1
        self.previous_time = current_time
        # self.update() #垂直同期切ったままこれ使うとfps500くらいになるので注意
        
    # ‑‑‑-------- Interaction ------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space: #スペースキー
            self.is_paused = not self.is_paused
            if self.is_paused: #止まり始めた時間を記録
                self._status_callback(text = "一時停止")
                self.pause_start_time = time.perf_counter()
            else:
                # 停止してた時間を足す
                if self.pause_start_time is not None:
                    self.total_paused_time += time.perf_counter() - self.pause_start_time
                    self.pause_start_time = None
                    self._status_callback(text = "再開")
                    self.update()
            return
        
        mods = event.modifiers()
        press_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        direction = self.cam_target - self.cam_posi  # 注視ベクトル
        if not press_ctrl:
            sensitivity = 0.03  # 移動・回転の感度
            horizontal_dist = np.sqrt((self.cam_target.x - self.cam_posi.x) ** 2 + (self.cam_target.y - self.cam_posi.y) ** 2)
            dz = self.cam_target.z - self.cam_posi.z
            elevation_angle = np.arctan2(dz, horizontal_dist)  # 水平面上の角度
            
            if event.key() == Qt.Key.Key_Up:
                elevation_angle += sensitivity
                self.cam_target.z = self.cam_posi.z + horizontal_dist * np.tan(elevation_angle)
                self.cam_target.z = max(0.0, min(self.cam_target.z, 10.0)) # z座標limit

            elif event.key() == Qt.Key.Key_Down:
                elevation_angle -= sensitivity
                self.cam_target.z = self.cam_posi.z + horizontal_dist * np.tan(elevation_angle)
                self.cam_target.z = max(0.0, min(self.cam_target.z, 10.0)) # z座標limit

            elif event.key() == Qt.Key.Key_Left:
                # （Z軸周りの反時計回り）
                angle = sensitivity
                rot = glm.rotate(glm.mat4(1), angle, glm.vec3(0, 0, 1))
                dir_rotated = glm.vec3(rot * glm.vec4(direction, 1.0))
                self.cam_target = self.cam_posi + dir_rotated

            elif event.key() == Qt.Key.Key_Right:
                # （Z軸周りの時計回り）
                angle = -sensitivity
                rot = glm.rotate(glm.mat4(1), angle, glm.vec3(0, 0, 1))
                dir_rotated = glm.vec3(rot * glm.vec4(direction, 1.0))
                self.cam_target = self.cam_posi + dir_rotated
        else:
            # Ctrlキー押下中: カメラ位置と注視点をともに平行移動
            xy_direction = glm.normalize(glm.vec3(direction.x, direction.y, 0))
            sensitivity = 0.1
            forward_direction =  xy_direction * sensitivity
            left_direction = glm.vec3(-xy_direction.y, xy_direction.x, 0) * sensitivity
            if event.key() == Qt.Key.Key_Up:
                self.cam_posi += forward_direction
                self.cam_target += forward_direction
            elif event.key() == Qt.Key.Key_Down:
                self.cam_posi -= forward_direction
                self.cam_target -= forward_direction
            elif event.key() == Qt.Key.Key_Left:
                self.cam_posi += left_direction
                self.cam_target += left_direction
            elif event.key() == Qt.Key.Key_Right:   
                self.cam_posi -= left_direction
                self.cam_target -= left_direction
        
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        mods = event.modifiers()
        press_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if not press_ctrl:
            if self._ray_show:
                self._ray_show = False
            return
        # Ctrl 押下中：レイを更新
        x, y = float(event.position().x()), float(event.position().y())
        ro, rd = self._make_ray(x, y)
        P = _ray_hit_plane(ro, rd, plane_point=self.plane_point, plane_normal=self.plane_normal)
        
        if P is None:
            if self._ray_show:
                self._ray_show = False
            return
        else:
            self._ray_p0, self._ray_p1 = P, P+ self.plane_normal * 0.5
            if not self._ray_show:
                self._ray_show = True
        
    def mousePressEvent(self, event: QMouseEvent) -> None:
        try:
            mods = event.modifiers()
            press_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
            if not press_ctrl:
                return
            
            #以降はctrlキー押下時の処理
            x, y = float(event.position().x()), float(event.position().y()) #マウスのqt座標
            ro, rd = self._make_ray(x, y)
            
            if event.button() == Qt.MouseButton.LeftButton: #左クリック
                # === 生成 ===
                P = _ray_hit_plane(ro, rd, plane_point=self.plane_point, plane_normal=self.plane_normal)  # checkerboard面
                if P is None:
                    return
                
                # if glm.length(P - self.plane_point) > param_changable["checkerboard"]["length"]:
                if abs(P.x - self.plane_point.x) > param_changable["checkerboard"]["length"] or abs(P.y - self.plane_point.y) > param_changable["checkerboard"]["length"]:
                    # planeがz=0の場合だけでちゃんと機能する
                    print("hit point is too far from origin, skipping")
                    return
                # ----- Object3D ballを生成 -----
                r = float(self.radius)  # スライダーで更新される半径
                center = P+ self.plane_normal * r  # 平面上の点から半径分だけ上にずらす
                subdiv = 2 if r < 0.5 else 3
                vertices, tri_indices = get_oneball_vertices_faces(subdiv=subdiv, radius=r)

                # 名前を決める
                nickname = self.parent().name_input.toPlainText().strip()
                if nickname != "":
                    name= nickname
                else:
                    num_ball = sum(1 for obj in self.simbuff.objects if "ball" in obj.name) + len(self.appended_object)
                    name = f"ball{num_ball+1}"
                # 色を決める
                if self.randomize_appended_obj_color:
                    color = tuple(rngnp.random(3))
                else:
                    c = self.parent()._picked_color
                    color = (c.redF(), c.greenF(), c.blueF())  # float 0~1に変換
                
                ball = Object3D(
                    vertices=vertices,
                    tri_indices=tri_indices,
                    color=color,
                    posi=(center.x, center.y, center.z),
                    radius=r,
                    is_move=True,
                    name=name,
                )
                self.makeCurrent()
                ball.create_gpuBuffer()  # GPUバッファを生成
                # 次のpaintGLで追加するリストへ登録
                self.appended_object.append(ball)
                
            elif event.button() == Qt.MouseButton.RightButton: #右クリック
                # === 削除（レイ上で最初に当たる球） ===
                hit_idx = -1
                hit_t = float("inf")
                removed_obj = None

                for obj in self.simbuff.objects:
                    if "ball" not in getattr(obj, "name", ""):# 球のみが削除対象
                        continue
                    center = glm.vec3(*obj.position)
                    
                    if obj.radius is None: #ballには必ずradius属性があるが一応
                        radius = max(obj.scale.x, obj.scale.y, obj.scale.z) * 1.0
                    else:
                        radius = obj.radius
                        
                    t = _ray_hit_sphere(ro, rd, center, radius)
                    #tが最小のものを選ぶ
                    if t is not None and t < hit_t:
                        hit_t = t
                        hit_idx = obj.obj_id
                        removed_obj = obj
                if hit_idx >= 0:
                    self.makeCurrent()
                    removed_obj.destroy_gpuBuffer()
                    self.removed_object_idx.append(hit_idx)

        finally:
            self.update()
            
    def wheelEvent(self, event: QMouseEvent):
        angle = event.angleDelta().y()
        length = glm.length(self.cam_target - self.cam_posi) #  カメラと注視点の距離
        direc = glm.normalize(self.cam_target - self.cam_posi)  # 注視ベクトル
        # スクロールの感度（1段階で ±0.1 変化）
        sensitivity = 0.1 #大きいほどちょっとの操作でめちゃズームする
        delta = -sensitivity if angle < 0 else sensitivity
        length -= delta

        # 最小・最大ズーム制限
        length = max(0.1, min(length, 100.0))

        self.cam_posi = self.cam_target - direc * length
    # ‑‑‑-------- その他 ------------
    def _make_ray(self, x: float, y: float):
        """スクリーン座標(x,y)→ワールド空間のレイ(origin, dir)"""
        # ビューポート（OpenGL座標系は左下原点なのでY反転）
        w, h = self.width(), self.height()
        winx, winy = x, h - y
        viewport = glm.vec4(0, 0, w, h)

        # 近/遠平面上の点をGLMで逆射影　#qt座標→ワールド座標変換
        near_world = glm.unProject(glm.vec3(winx, winy, 0.0), self.view, self.proj, viewport)
        far_world  = glm.unProject(glm.vec3(winx, winy, 1.0), self.view, self.proj, viewport)
        #0.0で近接面, 1.0で遠方面 それぞれglm.perspectiveで設定ずみ
        ro = near_world # ray origin
        rd = glm.normalize(far_world - near_world) #単位方向ベクトル ray direction
        return ro, rd
    
    def FpsTimer(self):
        # 1秒ごとの増分を計算
        fps = self.frameCount - self.previous_frameCount
        self.previous_frameCount = self.frameCount
        self.fpsUpdated.emit(fps) #シグナルをmain windowに送信
    
