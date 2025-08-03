# GLWidget.py

from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QFont, QPen, QColor, QPainterPath
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from OpenGL import GL
import glm
from typing import List, Optional
from pathlib import Path
import time

from tools import xp, np, load_shader, create_periodic_timer, param, param_changable, working_dir  # CuPy/NumPy, 各種ユーティリティ, ハイパーパラメータ
from create_obj import Object3D, create_boxes, create_axes  # オブジェクト生成はここに分離
from movie_ffepeg import MovieFFmpeg

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
        self.objects: List[Object3D] = []        # 描画対象オブジェクトリスト
        self.is_saving: bool = False             # 動画保存フラグ
        self.total_frame: int = 0                # 保存する総フレーム数
        self.aspect: float = 1.0                 # ウィンドウアスペクト比
        self.show_labels: bool = True  # ラベル表示フラグ
        self.radius: float = 1.0  # 半径（スライダーで調整，未使用）
        
        self.previous_frameCount:int = 0
        self.record_fps_timer = create_periodic_timer(self, self.FpsTimer, 1000)
        
        self.ctrl_fps_timer = create_periodic_timer(self, self.update, max(5, 1000//int(param_changable["fps"])))

    def initializeGL(self) -> None:
        """
        OpenGL初期化処理。シェーダコンパイル、オブジェクト生成、背景色設定、動画保存準備など。
        """
        # --- シェーダプログラム読み込み・コンパイル ---
        vert_src = load_shader(working_dir/param.shader.vert)
        frag_src = load_shader(working_dir/param.shader.frag)
        self.prog = GL.glCreateProgram()
        for src, stype in [(vert_src, GL.GL_VERTEX_SHADER), (frag_src, GL.GL_FRAGMENT_SHADER)]:
            s = GL.glCreateShader(stype)
            GL.glShaderSource(s, src)
            GL.glCompileShader(s)
            if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
            GL.glAttachShader(self.prog, s)
        GL.glLinkProgram(self.prog)
        if not GL.glGetProgramiv(self.prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(self.prog).decode())

        # --- 3Dオブジェクト生成は create_obj.pyで ---
        self.box = create_boxes()
        self.axes = create_axes()
        self.objects = self.box + self.axes

        GL.glEnable(GL.GL_DEPTH_TEST)

        # --- 動画保存用ffmpeg準備 ---
        self.is_saving = bool(param.movie.is_saving)
        self.frameCount = 0
        self.ffmpeg = MovieFFmpeg(self.width(), self.height())
        if self.is_saving:
            self.resizeGL(self.width(), self.height())  # 初期化
            
        self.start_time = time.perf_counter()  # 描画開始時刻
        self.previous_time = self.start_time

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
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.prog)
        GL.glClearColor(*param_changable["bg_color"]) #背景色

        # --- ビュー・プロジェクション行列の生成 ---
        view = glm.lookAt(glm.vec3(2,-2,2), glm.vec3(0,0,0), glm.vec3(0,0,1)) #カメラ位置，注視点， 上方向
        proj = glm.perspective(glm.radians(param_changable["fov"]), self.aspect, 0.01, 100.0) #視野角， アスペクト比，近接面，遠方面

        # --- uniformロケーション取得 ---
        uModelLoc = GL.glGetUniformLocation(self.prog, "uModel")
        uViewLoc  = GL.glGetUniformLocation(self.prog, "uView")
        uProjLoc  = GL.glGetUniformLocation(self.prog, "uProj")
        uColorLoc = GL.glGetUniformLocation(self.prog, "uColor")

        # --- ビュー・プロジェクション行列をGPUへ送信 ---
        GL.glUniformMatrix4fv(uViewLoc, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(uProjLoc, 1, False, glm.value_ptr(proj))

        current_time = time.perf_counter()
        t = current_time - self.start_time  # 経過時間 [秒]
        dt_frame = current_time - self.previous_time  # 前フレームからの経過時間 [秒]
        
        # --- オブジェクトの描画 ---
        for obj in self.objects:
            obj.update(t)
            obj.draw(self.prog, uModelLoc, uColorLoc, xp=xp, np=np)  # CuPy/NumPy両対応

        # --- QPainterでラベル描画 ---
        if self.show_labels:
            painter = QPainter(self)
            font = QFont("sans-serif", 16, QFont.Weight.Bold)
            painter.setFont(font)
            for obj in self.objects:
                pos = obj.localframe_to_window(view, proj, (self.width(), self.height()))
                r, g, b = [int(c*255) for c in obj.color]
                path = QPainterPath()
                path.addText(pos[0], pos[1], painter.font(), obj.name)
                painter.setPen(QPen(Qt.GlobalColor.white, 1.5))
                painter.drawPath(path)
                painter.setPen(QPen(QColor(r,g,b), 0.6))
                painter.fillPath(path, QColor(r,g,b))
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
