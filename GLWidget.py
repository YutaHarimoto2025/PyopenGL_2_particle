# GLWidget.py

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QPen, QColor, QPainterPath, QMouseEvent
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from OpenGL import GL
from OpenGL.GLU import gluUnProject
import glm
from typing import List, Optional
from pathlib import Path
import time

from tools import xp, np, create_periodic_timer, param, param_changable, working_dir  # CuPy/NumPy, 各種ユーティリティ, ハイパーパラメータ
from graphic_tools import load_shader, compute_normals, _ray_hit_plane, _ray_hit_sphere  # シェーダ読み込み、法線計算、レイキャスト
from create_obj import create_boxes, create_axes, get_oneball_vertices_faces  # オブジェクト生成はここに分離
from object3d import Object3D  # 3Dオブジェクト定義
from movie_ffepeg import MovieFFmpeg
from physics import Physics  # 物理シミュレーションデータ
from rendering import Renderer

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
        self.radius: float = 1.0  # 半径（スライダーで調整）
        
        self.previous_frameCount:int = 0
        
        self.appended_object = []
        self.removed_object_idx = []

    def initializeGL(self) -> None:
        """
        OpenGL初期化処理。シェーダコンパイル、オブジェクト生成、背景色設定、動画保存準備など。
        """
        # --- シェーダプログラム読み込み・コンパイル ---
        self.renderer = Renderer()
        self.renderer.init_checkerboard() 
        
        # --- 動画保存用ffmpeg準備 ---
        self.is_saving = bool(param.is_saving)
        self.frameCount = 0
        if self.is_saving:
            self.ffmpeg = MovieFFmpeg(self.width(), self.height())
        
        self.phys = Physics(self.is_saving)  # 物理シミュレーションデータ
        self.phys.start_stepping()  # シミュレーションスレッド開始
        self._status_callback() # 初期状態のステータスを表示
        
        self.start_time = time.perf_counter()  # 描画開始時刻
        self.previous_time = self.start_time
        self.record_fps_timer = create_periodic_timer(self, self.FpsTimer, 1000)
        self.ctrl_fps_timer = create_periodic_timer(self, self.update, max(5, 1000//int(param_changable["fps"])))

    def resizeGL(self, w: int, h: int) -> None:
        """
        ウィンドウサイズ変更時に呼ばれる。OpenGLビューポートも更新。
        """
        GL.glViewport(0, 0, w, h)
        self.aspect = w / h if h > 0 else 1

    def paintGL(self) -> None:
        """
        3Dシーンの描画、QPainterによるラベル描画、動画保存処理。
        """
        GL.glClearColor(*param_changable["bg_color"]) #背景色
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # 透視投影行列
        cam_posi = glm.vec3(2, -2, 2)  # カメラ位置
        self.view = glm.lookAt(cam_posi, glm.vec3(0,0,0), glm.vec3(0,0,1)) #カメラ位置，注視点， 上方向
        self.proj = glm.perspective(glm.radians(param_changable["fov"]), self.aspect, 0.01, 100.0) #視野角， アスペクト比，近接面，遠方面
        
        # 正射影
        # cam_posi = glm.vec3(0, 0, 10)
        # self.view = glm.lookAt(cam_posi, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # self.proj = glm.ortho(-5.0, 5.0, -5.0, 5.0, 0.01, 100.0)
        
        self.renderer.set_common(cam_posi, self.view, self.proj)
        self.renderer.draw_checkerboard(self.view, self.proj) 

        current_time = time.perf_counter()
        t = current_time - self.start_time  # 経過時間 [秒]
        dt_frame = current_time - self.previous_time  # 前フレームからの経過時間 [秒]
        # print(dt_frame)
        
        # --- シミュレーション更新 ---
        self.phys.update_objects(t, dt_frame, appended=self.appended_object, removed_ids=self.removed_object_idx)
        if self.appended_object:
            self._status_callback(text = "オブジェクト追加しました")
            self.appended_object.clear()
        if self.removed_object_idx:
            self._status_callback(text = "オブジェクト削除しました")
            self.removed_object_idx.clear()

        # --- オブジェクトの描画 ---
        for obj in self.phys.objects:
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
            font = QFont("Noto Sans CJK JP", 16, QFont.Weight.Normal)
            painter.setFont(font)
            for obj in self.phys.objects:
                pos = obj.localframe_to_window(self.view, self.proj, (self.width(), self.height()))
                r, g, b, a = [int(c*255) for c in obj.color]
                path = QPainterPath()
                path.addText(pos[0], pos[1], painter.font(), obj.name)
                # painter.setPen(QPen(Qt.GlobalColor.white, 3))
                # painter.drawPath(path) # テキストの輪郭

                painter.fillPath(path, QColor(r,g,b))#中を塗りつぶす
            painter.end()

        # --- フレームカウント管理・動画保存処理 ---
        if self.is_saving:
            self.ffmpeg.step(self.frameCount)
        
        self.frameCount += 1
        self.previous_time = current_time
        # self.update() #垂直同期切ったままこれ使うとfps500くらいになるので注意
        
    def FpsTimer(self):
        # 1秒ごとの増分を計算
        fps = self.frameCount - self.previous_frameCount
        self.previous_frameCount = self.frameCount
        self.fpsUpdated.emit(fps) #シグナルをmain windowに送信
        
    # ‑‑‑ Interaction
    def mousePressEvent(self, event: QMouseEvent) -> None:
        try:
            mods = event.modifiers()
            is_ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)

            if not is_ctrl:
                return
            
            #以降はctrlキー押下時の処理
            x, y = float(event.position().x()), float(event.position().y()) #マウスのqt座標
            ro, rd = self._make_ray(x, y)
            
            if event.button() == Qt.MouseButton.LeftButton: #左クリック
                # === 生成 ===
                plane_point  = glm.vec3(0,0,0)
                plane_normal = glm.vec3(0,0,1)
                P = _ray_hit_plane(ro, rd, plane_point=plane_point, plane_normal=plane_normal)  # checkerboard面
                if P is None:
                    return
                r = float(self.radius)  # スライダーで更新される半径
                center = P+ plane_normal * r  # 平面上の点から半径分だけ上にずらす

                vertices, tri_indices = get_oneball_vertices_faces(subdiv=2, radius=r)

                # Object3D を生成して配置（動かさない）
                ball = Object3D(
                    vertices=vertices,
                    tri_indices=tri_indices,
                    color=(1.0, 0.5, 1.0),
                    posi=(center.x, center.y, center.z),
                    radius=r,
                    is_move=True,
                    name="ball_appended",
                )
                # 次のpaintGLで追加するリストへ登録
                self.appended_object.append(ball)
                
            elif event.button() == Qt.MouseButton.RightButton: #右クリック
                # === 削除（レイ上で最初に当たる球） ===
                hit_idx = -1
                hit_t = float("inf")

                for obj in enumerate(self.phys.objects):
                    if getattr(obj, "name", "") != "ball":
                        continue
                    # 球のみが削除対象
                    center = glm.vec3(*obj.position)
                    
                    if radius is None:
                        radius = max(obj.scale.x, obj.scale.y, obj.scale.z) * 1.0
                    else:
                        radius = obj.radius
                        
                    t = _ray_hit_sphere(ro, rd, center, radius)
                    #tが最小のものを選ぶ
                    if t is not None and t < hit_t:
                        hit_t = t
                        hit_idx = obj.obj_id

                if hit_idx >= 0:
                    self.removed_object_idx.append(hit_idx)

        finally:
            self.update()
    
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
