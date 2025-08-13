from OpenGL import GL
import glm
from typing import Optional, Tuple, Any
import math

from tools import xp, np, xpFloat, xpInt, npFloat, npInt
from graphic_tools import compute_normals, GLGeometry, compute_uvs, load_texture, seam_split

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
        uv_mode: str = "auto",  #UVのモード指定（auto/box/spherical/cylindrical）
        posi: Tuple[float, float, float] = (0, 0, 0), # 世界座標
        rot: glm.quat|None=None,
        color: Tuple[float, float, float, float] = (1, 1, 1, 1), #白
        scale: Tuple[float, float, float] = (1, 1, 1), # ローカル座標軸に沿って
        radius: float|None = None, #半径（指定があれば）
        name_posi_local: Tuple[float, float, float] = (0, 0, 0), #ローカル座標
        is_move: bool = True,
        name: str = "unnamed",
        texture_path:str = None
    ) -> None:
        
        self.vertices = np.asarray(vertices)
        self.line_indices = np.asarray(line_indices) if line_indices is not None else np.asarray(npInt([]))
        self.tri_indices = np.asarray(tri_indices) if tri_indices is not None else np.asarray(npInt([]))
        # print(name, color, posi)
        self.normals = compute_normals(self.vertices, self.tri_indices.reshape(-1, 3)) if len(self.tri_indices) > 0 else np.asarray(npFloat([])) # 法線ベクトル計算

        # --------- UV（未指定なら共通ロジックで自動生成）----------
        if texture_path is not None: # UVはテクスチャを使うときだけ生成
            if uvs is None or (hasattr(uvs, "size") and uvs.size == 0):
                try:
                    tri = self.tri_indices.reshape(-1, 3) if len(self.tri_indices) > 0 else None
                    self.uvs = compute_uvs(self.vertices, tri_indices=tri,
                                        normals=self.normals if self.normals.size else None,
                                        mode=uv_mode)
                except Exception:
                    # 予防的フォールバック（最低限の0埋め）
                    self.uvs = np.zeros((len(self.vertices), 2), dtype=np.float32)
            else:
                self.uvs = npFloat(uvs)
                
            if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
                raise ValueError(f"uvs must have shape (N,2), but got {self.uvs.shape}")
            if self.uvs.shape[0] not in (0, self.vertices.shape[0]):
                raise ValueError(f"uv count {self.uvs.shape[0]} does not match vertex count {self.vertices.shape[0]}")
            
            if self.tri_indices.size > 0:
                # 球/円柱のときだけでもOK。まずは常にU方向の継ぎ目対策を適用して問題なし。
                self.vertices, self.normals, self.uvs, self.tri_indices = seam_split(
                    self.vertices, self.normals, self.uvs, self.tri_indices
                )
        else:
            self.uvs = None  # UV不要
        # ----------------------------------------------------------
        self.update_texture(texture_path)  # テクスチャの更新
        self.color = (*color, 1.0) if len(color) == 3 else color if len(color) == 4 else (_ for _ in ()).throw(ValueError("Color must be a tuple of 3 or 4 floats"))

        self.name = name
        self.name_posi_local = name_posi_local
        self.position = glm.vec3(*posi)
        self.rotation = rot if rot is not None else glm.quat()  # クォータニオン
        self.scale = glm.vec3(*scale)
        if radius is not None:
            self.radius = radius
        elif "ball" in self.name:
            raise ValueError("Ball objects must specify a radius !!")
        self.is_move = is_move  # 動くかどうか
        self.obj_id: int | None = None #SimBufferで使う
        
        # 初期のモデル行列　動かないものはこれを使いまわす
        self.model_mat = glm.mat4(1)
        self.update_model_matrix(init_flag=True)  # 初期化時にモデル行列を計算
        
        self.gpu_ready = False
    
    def create_gpuBuffer(self):
        if self.gpu_ready: return
        self.geo = GLGeometry()
        # 位置
        self.geo.add_array(0, self.vertices, 3)
        # 法線（あれば）
        if self.normals.size > 0:
            self.geo.add_array(1, self.normals, 3)
        # UV（保持していれば）
        if self.uvs is not None and self.uvs.size > 0:
            self.geo.add_array(2, self.uvs, 2)
        # インデックス
        if self.line_indices.size > 0:
            self.geo.set_elements("lines", self.line_indices)
        if self.tri_indices.size > 0:
            self.geo.set_elements("tris",  self.tri_indices)
        self.gpu_ready = True
    
    def destroy_gpuBuffer(self):
        if not self.gpu_ready: return
        try:
            # 先に VAO 以外
            for _, buf in list(self.geo.buffers.items()):
                GL.glDeleteBuffers(1, [buf])
            # VAO
            GL.glDeleteVertexArrays(1, [self.geo.vao])
        finally:
            self.geo = None
            self.gpu_ready = False
        
    def update_texture(self, texture_path: str|None) -> None:
        if texture_path is not None:
            texture_id_temp = load_texture(texture_path)  # GLコンテキスト有効時に呼ぶ
            if texture_id_temp is not None:  # None や 0 でなければ成功
                self.texture_id = texture_id_temp
                self.use_tex = bool(self.texture_id) #ロード成功したらTrue
            else:
                print(f"Failed to load texture from {texture_path}")
                if not hasattr(self, "texture_id"): #初回だけ
                    self.texture_id = None
                    self.use_tex = False
        else:
            self.texture_id = None
            self.use_tex = False

    def update_posi_rot(self, dt: float) -> None:
        if not self.is_move:
            return
        else:
            # self.position += glm.vec3(0, 0, dt * 0.1)  # Z軸方向に移動
            
            d_angle = glm.radians(dt * 60)
            dq = glm.angleAxis(d_angle, glm.vec3(0, 0, 1)) # 回転軸はZ軸
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