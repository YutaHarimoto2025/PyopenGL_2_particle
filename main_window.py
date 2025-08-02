# main_window.py

import os,sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QStatusBar,
    QLabel, QSlider, QDockWidget, QListWidget, QTextEdit, QWidget
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer

from GLWidget import GLWidget  # 別ファイルで定義するGLWidgetをインポート
from tools import param, update_param_changable  # ハイパーパラメータを読み込む

# Ensure X11 platform for stability
# os.environ.setdefault("QT_QPA_PLATFORM", "xcb") #GLSLベースだとこれは使っちゃだめ
os.environ.setdefault("XDG_SESSION_TYPE", "x11")
os.environ.setdefault("GDK_BACKEND", "x11")

# class MainWindow(QMainWindow):
#     """
#     アプリ全体のウィンドウ。中央にOpenGL描画(GLWidget)を配置し、ラベル入力・オブジェクトリスト・
#     ステータスバー・ツールバー等のUIを一括管理する。
#     """

#     def __init__(self) -> None:
#         super().__init__()
#         self.setWindowTitle("2D 円クリック生成・削除 (Refactored)")
#         self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

#         # 中央ウィジェットにGLWidgetを配置
#         self.gl = GLWidget(self._update_status)
#         self.setCentralWidget(self.gl)

#         # ステータスバー設置
#         self.status = QStatusBar()
#         self.setStatusBar(self.status)

#         # メニューバー設置
#         view_menu = self.menuBar().addMenu("表示")
#         clear_act = QAction("全削除", self)
#         clear_act.triggered.connect(self._clear_objects)
#         view_menu.addAction(clear_act)

#         # ツールバー（半径スライダー・ラベル表示切替ボタン）
#         tb = QToolBar("操作")
#         self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
#         radius_slider = QSlider(Qt.Orientation.Horizontal)
#         radius_slider.setRange(1, 100)
#         radius_slider.setValue(int(self.gl.radius * 100))
#         radius_slider.valueChanged.connect(self._radius_changed)
#         tb.addWidget(QLabel("半径"))
#         tb.addWidget(radius_slider)
        
#         # ラベル表示トグル
#         label_toggle = QAction("ラベル表示", self)
#         label_toggle.setCheckable(True)
#         label_toggle.setChecked(True)
#         label_toggle.triggered.connect(self._toggle_labels)
#         tb.addAction(label_toggle)

#         # 左ドック: オブジェクト一覧リスト
#         self.list_widget = QListWidget()
#         self.left1_dock = self._create_dock("オブジェクト一覧", self.list_widget, Qt.DockWidgetArea.LeftDockWidgetArea)
#         self.left1_dock.setFixedWidth(200)  # 横幅固定

#         # 右ドック: 名前入力欄
#         self.name_input = QTextEdit()
#         self.name_input.setPlaceholderText("ここに円のラベルを入力…")
#         self.left2_dock = self._create_dock("名前入力", self.name_input, Qt.DockWidgetArea.RightDockWidgetArea)

#         # ドックの縦分割
#         self.splitDockWidget(self.left1_dock, self.left2_dock, Qt.Orientation.Vertical)
#         self.resizeDocks(
#             [self.left1_dock, self.left2_dock],
#             [3, 1],
#             Qt.Orientation.Vertical
#         )
#         # self.show()  # 呼び出し側でshow()する想定

#     def _create_dock(self, title: str, widget: QWidget, area: Qt.DockWidgetArea) -> QDockWidget:
#         """
#         サイドドックを生成し、指定エリアに追加する。
#         """
#         dock = QDockWidget(title, self)
#         dock.setWidget(widget)
#         self.addDockWidget(area, dock)
#         return dock

#     def _radius_changed(self, value: int) -> None:
#         """
#         半径スライダー変更時の処理。GLWidgetの半径値を更新し、ステータス表示も更新。
#         """
#         self.gl.radius = value / 100.0
#         self.status.showMessage(f"半径: {self.gl.radius:.2f}")

#     def _clear_objects(self) -> None:
#         """
#         すべてのオブジェクトを削除し、UIもリセット。
#         """
#         self.gl.objects.clear()
#         self.gl.update()
#         self.list_widget.clear()
#         self.status.showMessage("すべてのオブジェクトを削除しました")

#     def _update_status(self, x: float, y: float, count: int) -> None:
#         """
#         GLWidgetからのコールバックで座標やオブジェクト数を表示・リスト更新。
#         """
#         self.status.showMessage(f"クリック位置: ({x:.2f}, {y:.2f}) | オブジェクト数: {count}")
#         self.list_widget.clear()
#         for i, obj in enumerate(self.gl.objects, start=1):
#             self.list_widget.addItem(f"{i}: ({obj.x:.2f}, {obj.y:.2f}) r={obj.r:.2f} '{obj.label}'")
            
#     def _toggle_labels(self, checked: bool) -> None:
#         """
#         ラベル表示ON/OFFの切り替え。
#         """
#         self.gl.show_labels = checked
#         self.gl.update()  # 再描画

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D座標変換デモ")
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.resize(800, 600)
        self.glWidget = GLWidget(self)
        self.setCentralWidget(self.glWidget)
        self.param_timer = QTimer(self)
        self.param_timer.timeout.connect(update_param_changable)
        self.param_timer.start(1000)  # 1000ミリ秒 = 1秒
# --- エントリーポイント ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(param.window_width, param.window_height)
    window.show()
    sys.exit(app.exec())
