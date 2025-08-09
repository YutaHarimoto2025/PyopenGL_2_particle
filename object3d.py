from OpenGL import GL
import glm
from typing import Optional, Tuple, Any
import math

from tools import xp, np, xpFloat, xpInt, npFloat, npInt
from graphic_tools import compute_normals  

class Object3D:
    """
    3Dオブジェクト（直方体、座標軸など）の頂点・インデックス・色・モデル変換などを管理。
    OpenGLバッファの生成・描画・座標変換も担当。
    """

    def __init__(
        self,
        vertices: np.ndarray|list,
        line_indices: np.ndarray|list|None = None,
        tri_indices: np.ndarray|list|None = None,
        posi: Tuple[float, float, float] = (0, 0, 0), # 世界座標
        rot: glm.quat|None=None,
        color: Tuple[float, float, float, float] = (1, 1, 1, 1), #白
        scale: Tuple[float, float, float] = (1, 1, 1), # ローカル座標軸に沿って
        name_posi_local: Tuple[float, float, float] = (0, 0, 0), #ローカル座標
        is_move: bool = True,
        name: str = "",
    ) -> None:
        # --- CuPy/NumPy配列型で受け取りfloat32/uint32で型変換 ---
        self.vertices = np.asarray(vertices)
        self.line_indices = np.asarray(line_indices) if line_indices is not None else np.asarray(npInt([]))
        self.tri_indices = np.asarray(tri_indices) if tri_indices is not None else np.asarray(npInt([]))
        print(name, color, posi)
        self.normals = compute_normals(self.vertices, self.tri_indices.reshape(-1, 3)) if len(self.tri_indices) > 0 else np.asarray(npFloat([])) # 法線ベクトル計算
        if len(color) == 3:
            self.color = (*color, 1.0)
        else:
            self.color = color
        self.name = name
        self.name_posi_local = name_posi_local
        self.position = glm.vec3(*posi)
        self.rotation = rot if rot is not None else glm.quat()  # クォータニオン
        self.scale = glm.vec3(*scale)
        self.is_move = is_move  # 動くかどうか
        
        # 初期のモデル行列　動かないものはこれを使いまわす
        self.model_mat = glm.mat4(1)
        self.update_model_matrix(init_flag=True)  # 初期化時にモデル行列を計算
        
        # --- OpenGLバッファ生成 ---
        # 頂点配列（必須）
        self.vbo = GL.glGenBuffers(1)
        # 法線配列
        self.vbo_normal = GL.glGenBuffers(1) if hasattr(self, "normals") and self.normals is not None and len(self.normals) > 0 else None
        # UV配列（必要なら）
        self.vbo_uv = GL.glGenBuffers(1) if hasattr(self, "uvs") and self.uvs is not None and len(self.uvs) > 0 else None
        self.ebo = GL.glGenBuffers(1)
        self.vao = GL.glGenVertexArrays(1)

        GL.glBindVertexArray(self.vao)

        # 位置バッファ
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)

        # 法線バッファ
        if self.vbo_normal is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_normal)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, False, 0, None)

        # UVバッファ（必要なら）
        if self.vbo_uv is not None:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo_uv)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, self.uvs.nbytes, self.uvs, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(2)
            GL.glVertexAttribPointer(2, 2, GL.GL_FLOAT, False, 0, None)

        # EBO (index buffer)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # indexは後で描画時に切替えても良いし、どちらか初期転送でも良い
        if self.line_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_indices.nbytes, self.line_indices, GL.GL_STATIC_DRAW)
        elif self.tri_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.tri_indices.nbytes, self.tri_indices, GL.GL_STATIC_DRAW)

        GL.glBindVertexArray(0)

    def update_posi_rot(self, dt: float) -> None:
        if not self.is_move:
            return
        else:
            # self.position += glm.vec3(0, 0, dt * 0.1)  # Z軸方向に移動
            
            d_angle = glm.radians(dt * 60)
            dq = glm.angleAxis(d_angle, glm.vec3(0, 1, 0)) # 回転軸はY軸
            self.rotation = dq * self.rotation  # クォータニオンの積で回転　演算は非可換
    
    def update_model_matrix(self, init_flag=False) -> None:
        if init_flag is False and not self.is_move:
            return
        else:
            self.model_mat = (
                    glm.translate(glm.mat4(1), self.position)
                    * glm.mat4_cast(self.rotation)
                    * glm.scale(glm.mat4(1), self.scale)
                )

    def draw(self, prog: int, uModelLoc: int, uColorLoc: int, xp=xp, np=np) -> None:
        """
        OpenGL描画処理。バッファ転送や属性設定も含む。
        xp/npを引数で指定しCuPy/NumPy両対応。
        """
        GL.glUniformMatrix4fv(uModelLoc, 1, False, glm.value_ptr(self.model_mat))
        GL.glUniform4f(uColorLoc, *self.color)
        GL.glBindVertexArray(self.vao)

        # --- LINE描画 ---
        if self.line_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_indices.nbytes, self.line_indices, GL.GL_STATIC_DRAW)
            GL.glDrawElements(GL.GL_LINES, len(self.line_indices), GL.GL_UNSIGNED_INT, None)
        # --- TRI描画 ---
        if self.tri_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.tri_indices.nbytes, self.tri_indices, GL.GL_STATIC_DRAW)
            GL.glDrawElements(GL.GL_TRIANGLES, len(self.tri_indices), GL.GL_UNSIGNED_INT, None)

        GL.glBindVertexArray(0)

    def localframe_to_window(
        self, view: glm.mat4, proj: glm.mat4, viewport_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        ローカル座標系のラベル位置をウィンドウ座標系に変換して返す（ラベル描画用）
        """
        posi = glm.vec4(self.name_posi_local, 1)
        world = self.model_mat * posi
        clip = proj * view * world
        ndc = glm.vec3(clip.x, clip.y, clip.z) / clip.w
        w, h = viewport_size
        x_win = int((ndc.x + 1) * 0.5 * w)
        y_win = int((1 - (ndc.y + 1) * 0.5) * h)
        return (x_win, y_win)