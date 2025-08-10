from OpenGL import GL
import glm
from typing import Optional, Tuple, Any
import math

from tools import xp, np, xpFloat, xpInt, npFloat, npInt
from graphic_tools import compute_normals, GLGeometry

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
        uvs: np.ndarray|None= np.empty((0, 2), dtype=np.float32),
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
        # print(name, color, posi)
        self.normals = compute_normals(self.vertices, self.tri_indices.reshape(-1, 3)) if len(self.tri_indices) > 0 else np.asarray(npFloat([])) # 法線ベクトル計算
        self.uvs = npFloat(uvs)
        if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
            raise ValueError(f"uvs must have shape (N,2), but got {self.uvs.shape}")
        if self.uvs.shape[0] not in (0, self.vertices.shape[0]):
            raise ValueError(f"uv count {self.uvs.shape[0]} does not match vertex count {self.vertices.shape[0]}")
        self.color = (*color, 1.0) if len(color) == 3 else color if len(color) == 4 else (_ for _ in ()).throw(ValueError("Color must be a tuple of 3 or 4 floats"))

        self.name = name
        self.name_posi_local = name_posi_local
        self.position = glm.vec3(*posi)
        self.rotation = rot if rot is not None else glm.quat()  # クォータニオン
        self.scale = glm.vec3(*scale)
        self.is_move = is_move  # 動くかどうか
        
        # 初期のモデル行列　動かないものはこれを使いまわす
        self.model_mat = glm.mat4(1)
        self.update_model_matrix(init_flag=True)  # 初期化時にモデル行列を計算
        
        ################ --- OpenGLバッファ生成 ---
        self.geo = GLGeometry()
        # 位置
        self.geo.add_array(0, self.vertices, 3)
        # 法線（あれば）
        if self.normals.size > 0:
            self.geo.add_array(1, self.normals, 3)
        # UV（保持していれば）
        if self.uvs.size > 0:
            self.geo.add_array(2, self.uvs, 2)
        # インデックス
        if self.line_indices.size > 0:
            self.geo.set_elements("lines", self.line_indices)
        if self.tri_indices.size > 0:
            self.geo.set_elements("tris",  self.tri_indices)
        
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

    def draw(self) -> None:
        """
        OpenGL描画処理。バッファ転送や属性設定も含む。
        xp/npを引数で指定しCuPy/NumPy両対応。
        """
        if self.line_indices.size > 0:
            self.geo.draw_elements(GL.GL_LINES, "lines")
        if self.tri_indices.size > 0:
            self.geo.draw_elements(GL.GL_TRIANGLES, "tris")

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