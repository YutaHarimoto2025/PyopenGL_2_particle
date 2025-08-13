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
from event_handler import EventHandler


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
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True) # ホバーイベントを受け取る
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus) # キーボードイベントを受け取るためにフォーカスを強制的に設定
        # --- 動画保存用ffmpeg準備 ---
        self.is_saving = bool(param.is_saving)
        self.frameCount = 0
        if self.is_saving:
            self.ffmpeg = MovieFFmpeg(self.width(), self.height())
        
        self.simbuff = SimBuffer(self.is_saving)  # 物理シミュレーションデータ
        self.simbuff.start_stepping()  # シミュレーションスレッド開始
        self._status_callback() # 初期状態のステータスを表示
        
        self.handler = EventHandler(self)

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
    
    # --- イベントハンドラ ---   
    def keyPressEvent(self, event):
        self.handler.handle_key_press(event)

    def mouseMoveEvent(self, event):
        self.handler.handle_mouse_move(event)

    def mousePressEvent(self, event):
        self.handler.handle_mouse_press(event)

    def wheelEvent(self, event):
        self.handler.handle_wheel(event)
        
    # 追加：ウィンドウ外に出た瞬間に回転停止
    def leaveEvent(self, event):
        # EventHandler 側のエッジ回転を止める
        self.handler._stop_edge_rotation()
        event.accept()

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
    
