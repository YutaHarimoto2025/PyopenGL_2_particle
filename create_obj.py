# create_obj.py

from OpenGL import GL
import glm
from typing import Optional, Tuple, Any
from tools import xp, np, xpFloat, xpInt, npFloat, npInt

class Object3D:
    """
    3Dオブジェクト（直方体、座標軸など）の頂点・インデックス・色・モデル変換などを管理。
    OpenGLバッファの生成・描画・座標変換も担当。
    """

    def __init__(
        self,
        vertices: np.ndarray|list,
        line_indices: np.ndarray|list,
        tri_indices: np.ndarray|list,
        color: Tuple[float, float, float],
        posi: Tuple[float, float, float] = (0, 0, 0), # 世界座標
        rot: glm.quat|None = None,
        scale: Tuple[float, float, float] = (1, 1, 1), # ローカル座標軸に沿って
        name_posi_local: Tuple[float, float, float] = (0, 0, 0), #ローカル座標
        is_move: bool = True,
        name: str = "",
    ) -> None:
        # --- CuPy/NumPy配列型で受け取りfloat32/uint32で型変換 ---
        self.vertices = np.asarray(vertices)
        self.line_indices = np.asarray(line_indices)
        self.tri_indices = np.asarray(tri_indices)
        self.color = color
        self.name = name
        self.name_posi_local = name_posi_local
        self.position = glm.vec3(*posi)
        self.rotation = rot if rot is not None else glm.quat()
        self.scale = glm.vec3(*scale)
        self.is_move = is_move  # 動くかどうか

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL.GL_STATIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        # EBOはLINE/TRI両方バッファ共有
        if self.line_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.line_indices.nbytes, self.line_indices, GL.GL_STATIC_DRAW)
        elif self.tri_indices.size > 0:
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.tri_indices.nbytes, self.tri_indices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0) # 位置ベクトルを有効化
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 0, None)
        GL.glBindVertexArray(0)
        
        # 初期のモデル行列　動かないものはこれを使いまわす
        self.model_mat = glm.mat4(1)
        self.update_model_matrix()  # 初期化時にモデル行列を計算

    def update_posi_rot(self, dt: float) -> None:
        if not self.is_move:
            return
        else:
            # self.position += glm.vec3(0, 0, dt * 0.1)  # Z軸方向に移動
            
            d_angle = glm.radians(dt * 60)
            dq = glm.angleAxis(d_angle, glm.vec3(0, 1, 0)) # 回転軸はY軸
            self.rotation = dq * self.rotation  # クォータニオンの積で回転　演算は非可換
    
    def update_model_matrix(self) -> None:
        if not self.is_move:
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
        GL.glUniform3f(uColorLoc, *self.color)
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

# -------------------------------------------------------
# 3Dオブジェクト生成用ファクトリ関数
# -------------------------------------------------------

def create_boxes() -> list:
    """
    直方体オブジェクトをxp(CuPy/NumPy)で生成し返す
    """
    obj_list = []
    
    vertices = npFloat(
        [
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5]
        ]
    )
    line_indices = npInt(
        [
        0,1, 1,2, 2,3, 3,0,    # -Z面
        4,5, 5,6, 6,7, 7,4,    # +Z面
        0,4, 1,5, 2,6, 3,7,    # 側面
        ]
    )
    tri_indices = npInt(
        [
        0,1,2, #0,2,3,  # -Z面 (例)
        # 4,5,6, 4,6,7, ...
        ]
    )
    color = (1.0, 1.0, 0.2) # 黄色
    
    obj_list.append(Object3D(vertices, line_indices, tri_indices, color, name="Box", name_posi_local=(0,0,0)))
    return obj_list

def create_axes() -> list:
    """
    X/Y/Z座標軸のオブジェクトをリストで返す
    """
    axes = [
        Object3D(
            vertices = npFloat([[0, 0, 0], [1, 0, 0]]),
            line_indices = npInt([0, 1]),
            tri_indices = npInt([]),
            color = (1, 0, 0),
            name = "X",
            name_posi_local = (1.2,0,0),
            is_move = False
        ),
        Object3D(
            vertices = npFloat([[0, 0, 0], [0, 1, 0]]),
            line_indices = npInt([0, 1]),
            tri_indices = npInt([]),
            color = (0, 1, 0),
            name = "Y",
            name_posi_local = (0,1.2,0),
            is_move = False
        ),
        Object3D(
            vertices = npFloat([[0, 0, 0], [0, 0, 1]]),
            line_indices = npInt([0, 1]),
            tri_indices = npInt([]),
            color = (0, 0, 1),
            name = "Z",
            name_posi_local = (0,0,1.2),
            is_move = False
        ),
    ]
    return axes
