import sys, os
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL


os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_XCB_GL_INTEGRATION"] = "xcb_glx"
os.environ["PYOPENGL_PLATFORM"] = "glx"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["GDK_BACKEND"] = "x11"

class EnvironmentChecker:
    """環境の健全性をチェックするクラス"""
    
    def __init__(self) -> None:
        self.python_version = sys.version
        self.qt_version = QT_VERSION_STR
        self.pyqt_version = PYQT_VERSION_STR

    def display_info(self) -> None:
        """現在のバージョン情報を出力する"""
        print(f"--- Environment Information ---")
        print(f"Python Version: {self.python_version}")
        print(f"Qt Version:     {self.qt_version}")
        print(f"PyQt6 Version:  {self.pyqt_version}")
        print(f"-------------------------------")
        print("モジュールのロードに成功しました。")

if __name__ == "__main__":
    try:
        checker = EnvironmentChecker()
        checker.display_info()
    except Exception as e:
        print(f"エラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)