# main_window.py

import os,sys, time
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QStatusBar,
    QLabel, QSlider, QDockWidget, QListWidget, QTextEdit, 
    QWidget, QColorDialog, QPushButton, QVBoxLayout,
    QHBoxLayout, QCheckBox, QToolButton
)
from PyQt6.QtGui import QAction, QSurfaceFormat, QColor
from PyQt6.QtCore import Qt, QTimer

from GLWidget import GLWidget  # 別ファイルで定義するGLWidgetをインポート
from tools import param, param_changable, update_param_changable, create_periodic_timer, rngnp  # ハイパーパラメータを読み込む
import os
import signal

# Ensure X11 platform for stability
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb") #GLSLベースだとこれは使っちゃだめ
# os.environ.setdefault("QT_OPENGL", "desktop")
os.environ.setdefault("XDG_SESSION_TYPE", "x11")
os.environ.setdefault("GDK_BACKEND", "x11")

class MainWindow(QMainWindow):
    """
    アプリ全体のウィンドウ。中央にOpenGL描画(GLWidget)を配置し、ラベル入力・オブジェクトリスト・
    ステータスバー・ツールバー等のUIを一括管理する。
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyopenGL Particle System")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

        # 中央ウィジェットにGLWidgetを配置
        self.gl = GLWidget(self._update_status) 
        self.setCentralWidget(self.gl)

        # ステータスバー設置
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        # ステータスバーにFPS表示ラベルを追加
        self.fpsLabel = QLabel("fps: 0", self)
        self.statusBar().addPermanentWidget(self.fpsLabel)
        self.gl.fpsUpdated.connect(self.onFpsUpdated)
        #fpsUpdatedシグナルはint型の値（fps）を持っていて、そのままonFpsUpdatedの引数に渡される

        # メニューバー設置
        view_menu = self.menuBar().addMenu("表示")
        clear_act = QAction("全削除", self)
        clear_act.triggered.connect(self._clear_objects)
        view_menu.addAction(clear_act)
        
        reset_act = QAction("リセット", self)
        reset_act.triggered.connect(self._reset_objects)
        view_menu.addAction(reset_act)

        # ツールバー（半径スライダー・ラベル表示切替ボタン）
        self.slider_radius_rate: int = 100
        tb = QToolBar("操作")
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        radius_slider = QSlider(Qt.Orientation.Horizontal)
        radius_slider.setRange(int(0.01 * self.slider_radius_rate), int(2 * self.slider_radius_rate))
        radius_slider.setValue(int(self.gl.radius * self.slider_radius_rate)) #デフォルト値
        radius_slider.valueChanged.connect(self._radius_changed)
        tb.addWidget(QLabel("半径"))
        tb.addWidget(radius_slider)
        
        # ラベル表示トグル
        label_toggle = QAction("ラベル表示", self) #QAction はwidggetに追加できない
        label_toggle.setCheckable(True)
        label_toggle.setChecked(False)
        label_toggle.triggered.connect(self._toggle_labels)
        tb.addAction(label_toggle)

        # 左上ドック: オブジェクト一覧リスト
        self.list_widget = QListWidget()
        self.left1_dock = self._create_dock("オブジェクト名: 座標", self.list_widget, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.left1_dock.setFixedWidth(250)  # 横幅固定

        # 左下ドック: 名前入力欄
        # 名前入力と色選択をまとめたパネルを作成
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.name_input = QTextEdit()
        self.name_input.setPlaceholderText("オブジェクトの名前を入力…")
        layout.addWidget(QLabel("名前"))
        layout.addWidget(self.name_input)

        self._picked_color = QColor(200, 200, 200)
        self.color_btn = QPushButton("色を選ぶ")
        self.color_btn.setStyleSheet(f"background-color: {self._picked_color.name()};")
        self.color_btn.clicked.connect(
            lambda: (
                lambda c=QColorDialog.getColor(self._picked_color, self, "色を選択"):
                    (setattr(self, "_picked_color", c),
                    self.color_btn.setStyleSheet(f"background-color: {c.name()};"))
                if c.isValid() else None
            )()
        )
        self.random_color_action = QAction("ランダム", self)
        self.random_color_action.setCheckable(True)
        self.random_color_action.setChecked(False)
        self.random_color_action.triggered.connect(self.get_color_rgb_tuple)
        
        random_color_btn = QToolButton()
        random_color_btn.setDefaultAction(self.random_color_action)

        # 色のラベルとボタン＋トグルを横並びに
        layout.addWidget(QLabel("色"))

        color_layout = QHBoxLayout()
        color_layout.addWidget(self.color_btn)
        color_layout.addWidget(random_color_btn)
        layout.addLayout(color_layout)

        # 既存の _create_dock を利用
        self.left2_dock = self._create_dock("追加オブジェクトの，名前 / 色", panel, Qt.DockWidgetArea.LeftDockWidgetArea)

        # ドックの縦分割
        self.splitDockWidget(self.left1_dock, self.left2_dock, Qt.Orientation.Vertical)
        self.resizeDocks(
            [self.left1_dock, self.left2_dock],
            [3, 1],
            Qt.Orientation.Vertical
        )
        
        # パラメータ更新タイマー
        self.update_param_timer = create_periodic_timer(self, self.param_updater, 1000)  # 1秒ごとに更新
    
    def get_color_rgb_tuple(self, checked:bool):
        self.gl.randomize_appended_obj_color = checked# ランダム色（0..1 の3成分）

    def _create_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
        """
        サイドドックを生成し、指定エリアに追加する。
        """
        dock = QDockWidget(title, self)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _radius_changed(self, value: int) -> None:
        """
        半径スライダー変更時の処理。GLWidgetの半径値を更新し、ステータス表示も更新。
        """
        self.gl.radius = float(value / self.slider_radius_rate)
        self.status.showMessage(f"半径: {self.gl.radius:.2f}")

    def _clear_objects(self) -> None:
        """
        すべてのオブジェクトを削除し、UIもリセット。
        """
        self.gl.simbuff.objects.clear()
        self.gl.update()
        self.list_widget.clear()
        self.status.showMessage("すべてのオブジェクトを削除しました")
        
    def _reset_objects(self) -> None:
        #球の位置はランダマイズされる
        if hasattr(self, "gl"):
            self.gl.setParent(None)
            self.gl.deleteLater()

        self.gl = GLWidget(self._update_status)  # 新しいGLWidgetインスタンス
        self.setCentralWidget(self.gl)
        self.status.showMessage("初期状態をランダマイズしてリセットしました")
        self.gl.update()    

    def _update_status(self, text:str=None) -> None:
        """
        GLWidgetからのコールバックで座標やオブジェクト数を表示・リスト更新。
        """
        # UI更新のため少し待つ
        if text is None:
            text=""
        self.status.showMessage(f"オブジェクト数: {len(self.gl.simbuff.objects)} | {text}") 
        self.list_widget.clear()
        for i, obj in enumerate(self.gl.simbuff.objects, start=1):
            self.list_widget.addItem(f"{obj.name}: ({obj.position.x:.2f}, {obj.position.y:.2f}, {obj.position.z:.2f})")
            
    def _toggle_labels(self, checked: bool) -> None:
        """
        ラベル表示ON/OFFの切り替え。
        """
        self.gl.show_labels = checked
        self.gl.update()  # 再描画
        
    def onFpsUpdated(self, fps):
        self.fpsLabel.setText(f"fps: {fps}")
        
    def param_updater(self):
        update_param_changable()  # パラメータの更新
        # 新しいfps値でタイマー再設定, 厳密に同じにはならない
        self.gl.ctrl_fps_timer.start(max(5, 1000 // int(param_changable["fps"])))
        
        if hasattr(self.gl.simbuff, "textural_ball") and self.gl.simbuff.textural_ball:
            self.gl.simbuff.textural_ball[0].update_texture(param_changable["ball_texture"])
        
        self.gl.handler.refresh_params()
        
    def closeEvent(self, event) -> None:
        print(f"PID {os.getpid()} killed") #kill -9
        os.kill(os.getpid(), signal.SIGKILL)
        event.accept()
            
        

# --- エントリーポイント ---
if __name__ == "__main__":
    format = QSurfaceFormat()
    format.setVersion(3, 0) # OpenGL 3.3 Core Profile
    format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    format.setDepthBufferSize(24) # 深度バッファのビット数
    format.setStencilBufferSize(8)
    # format.setSamples(4) #アンチエイリアスのサンプル数 オンにしたら動画保存でエラー
    format.setSwapInterval(0) # 垂直同期切ってfps爆速になる魔法，普段はディスプレイのfpsと同じがやや遅いか
    # format.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer) #ダブルバッファでなめらかに
    QSurfaceFormat.setDefaultFormat(format)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(param.window_width, param.window_height)
    window.show()
    sys.exit(app.exec())
