# GLWidget.py

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from OpenGL import GL
from OpenGL.GLU import gluUnProject
import glm
from typing import List, Optional
from pathlib import Path
import time

from tools import xp, np, create_periodic_timer, param, param_changable, working_dir  # CuPy/NumPy, 各種ユーティリティ, ハイパーパラメータ
from graphic_tools import load_shader
from create_obj import create_boxes, create_axes  # オブジェクト生成はここに分離
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
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.total_frame: int = 0                # 保存する総フレーム数
        self.aspect: float = 1.0                 # ウィンドウアスペクト比
        self.show_labels: bool = True  # ラベル表示フラグ
        self.radius: float = 1.0  # 半径（スライダーで調整，未使用）
        
        self.previous_frameCount:int = 0

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
        view = glm.lookAt(cam_posi, glm.vec3(0,0,0), glm.vec3(0,0,1)) #カメラ位置，注視点， 上方向
        proj = glm.perspective(glm.radians(param_changable["fov"]), self.aspect, 0.01, 100.0) #視野角， アスペクト比，近接面，遠方面
        
        # 正射影
        # cam_posi = glm.vec3(0, 0, 10)
        # view = glm.lookAt(cam_posi, glm.vec3(0,0,0), glm.vec3(0,1,0))
        # proj = glm.ortho(-5.0, 5.0, -5.0, 5.0, 0.01, 100.0)
        
        self.renderer.set_common(cam_posi, view, proj)
        self.renderer.draw_checkerboard(view, proj) 

        current_time = time.perf_counter()
        t = current_time - self.start_time  # 経過時間 [秒]
        dt_frame = current_time - self.previous_time  # 前フレームからの経過時間 [秒]
        # print(dt_frame)
        self.phys.update_objects(t, dt_frame)  # 物理シミュレーションの更新
        
        # --- オブジェクトの描画 ---
        for obj in self.phys.objects:
            self.renderer.set_each(obj.model_mat, obj.color)   # uModel / uNormalMatrix / uColor             # uColor
            obj.draw()  # CuPy/NumPy両対応

        # --- QPainterでラベル描画 ---
        if self.show_labels:
            painter = QPainter(self)
            font = QFont("Noto Sans CJK JP", 16, QFont.Weight.Normal)
            painter.setFont(font)
            for obj in self.phys.objects:
                pos = obj.localframe_to_window(view, proj, (self.width(), self.height()))
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
