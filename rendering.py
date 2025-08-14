import numpy as np
from OpenGL import GL
import glm
import ctypes

from tools import param_changable, param
from graphic_tools import load_shader, build_GLProgram, GLGeometry

# ---- uniform location を安全に取得（-1なら警告）
def get_uniform_loc(target, prog, name_set:set, prefix=""):
    for name in name_set:
        loc = GL.glGetUniformLocation(prog, name)
        if loc == -1:
            print(f"[warn] uniform '{name}' not found (optimized out or name mismatch)")
        setattr(target, f"{prefix}{name}", loc)

def apply_common_rendering_settings():
    GL.glEnable(GL.GL_DEPTH_TEST); GL.glDepthFunc(GL.GL_LESS); GL.glDepthMask(GL.GL_TRUE) 
    GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
    GL.glDisable(GL.GL_CULL_FACE); #スクリーン上で反時計回りが表面，ウラ面を描画しない
    GL.glCullFace(GL.GL_BACK) # #ウラ面を描画しない　＃GL_CULL_FACEのデフォ
    GL.glFrontFace(GL.GL_CCW) # 反時計回りが表面 #GL_CULL_FACEのデフォ
    GL.glDisable(GL.GL_BLEND)
    GL.glEnable(GL.GL_MULTISAMPLE) #アンチエイリアス
    
    # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL) # ポリゴンモードを確認（塗りつぶし）
    # GL.glEnable(GL.GL_DEPTH_CLAMP) # 深度クランプを有効化（オプション）

    # 透明なものを描く
    # GL.glEnable(GL.GL_BLEND)
    # GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
    # 必要なら奥→手前の順にソート描画
        
class ObjectRenderer:
    def __init__(self):
        self.prog = build_GLProgram(vert_filename=param.shader.vert, frag_filename=param.shader.frag)
        # ---- cache uniform locations
        uniform_name_set = {
            # 既存
            "uModel", "uView", "uProj", "uViewPos",
            "uLightPos", "uColor", "uShadingMode",
            "uRimStrength", "uGamma", "uNormalMatrix",
            # 追加（マテリアル/ライティング）
            "uDiffuse", "uLightColor", "uAmbient", "uSpecularStr", "uShininess",
            # 追加（UV/テクスチャ）
            "uUseTexture", "uTexIsSRGB", "uTex",
            "uUseUVChecker", "uUVCell", "uUVColor1", "uUVColor2",
        }
        get_uniform_loc(self, prog=self.prog, name_set=uniform_name_set)  

    def set_common(self, cam_posi, view, proj): #オブジェクト共通
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uView, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(self.uProj, 1, False, glm.value_ptr(proj))
        GL.glUniform3f(self.uViewPos, *cam_posi)
        GL.glUniform3f(self.uLightPos, *(5.0, 5.0, 5.0))
        GL.glUniform1i(self.uShadingMode, 2)    # 0:Phong / 1:Toon / 2:Phong+Rim
        GL.glUniform1f(self.uRimStrength, 0.2)  # 0~1 1に近いと輪郭が白くなる
        GL.glUniform1f(self.uGamma, 1.4)        # 1で補正なし，<1で暗くなる， >1で明るくなる
        
        # 追加：マテリアル/ライティングの既定値
        GL.glUniform1f(self.uDiffuse, 0.4) #拡散反射
        GL.glUniform3f(self.uLightColor, 1.0, 1.0, 1.0)
        GL.glUniform1f(self.uAmbient,     0.50) #ここでの設定がglslより優先される
        GL.glUniform1f(self.uSpecularStr, 0.25)
        GL.glUniform1f(self.uShininess,   32.0)

        # UV/テクスチャ関連の初期化 OFF
        GL.glUniform1i(self.uUseUVChecker, 0)
        GL.glUniform1f(self.uUVCell,       0.01)
        GL.glUniform3f(self.uUVColor1,     0.92, 0.92, 0.92)
        GL.glUniform3f(self.uUVColor2,     0.08, 0.08, 0.08)
        GL.glUniform1i(self.uTexIsSRGB,   1)

    def set_each(self, obj): #オブジェクトごと
        GL.glUseProgram(self.prog)
        model = obj.model_mat
        rgba = obj.color
        GL.glUniformMatrix4fv(self.uModel, 1, False, glm.value_ptr(model)) # uModel
        nmat = glm.mat3(glm.transpose(glm.inverse(glm.mat3(model)))) 
        GL.glUniformMatrix3fv(self.uNormalMatrix, 1, False, glm.value_ptr(nmat)) # uNormalMatrix非一様スケール対応
        GL.glUniform4f(self.uColor, *rgba) # uColor
        
        # テクスチャ
        if obj.use_tex:
            GL.glUniform1i(self.uUseTexture, 1)
        else:
            GL.glUniform1i(self.uUseTexture, 0)
        
    def draw(self, obj) -> None:
        """
        OpenGL描画処理。バッファ転送や属性設定も含む。
        xp/npを引数で指定しCuPy/NumPy両対応。
        """
        GL.glUseProgram(self.prog)
        if obj.use_tex and self.uTex != -1:
            # bind
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, obj.texture_id)
            # サンプラにユニット0を通知
            GL.glUniform1i(self.uTex, 0)

            # draw
            if obj.line_indices.size > 0:
                obj.geo.draw_elements(GL.GL_LINES, "lines")
            if obj.tri_indices.size > 0:
                obj.geo.draw_elements(GL.GL_TRIANGLES, "tris")

            # unbind
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        else:
            # 非テクスチャ物体
            if obj.line_indices.size > 0:
                obj.geo.draw_elements(GL.GL_LINES, "lines")
            if obj.tri_indices.size > 0:
                obj.geo.draw_elements(GL.GL_TRIANGLES, "tris")
                
class NonObjectRenderer:
    """
    オブジェクト以外の描画用クラス。
    例: Ray, Checkerboard, Planeなど。
    """
    def __init__(self, name:str, vertices:np.ndarray, indices:np.ndarray, draw_mode:str, additional_uniform_dict:dict):
        self.prog = build_GLProgram(vert_filename=f"{name}.vert", frag_filename=f"{name}.frag")
        base_uniform_set = {"uView", "uProj"}
        additional_uniform_set = set(additional_uniform_dict.copy().keys())
        get_uniform_loc(self, prog=self.prog, name_set=base_uniform_set|additional_uniform_set)
        self.geo = GLGeometry()
        self.geo.add_array(0, vertices, vertices.shape[1])  # layout(location=0) in vec2,3 ～ を想定 todo:柔軟に変更
        self.geo.set_elements(draw_mode, indices)
        
        self.draw_mode = draw_mode  # "tris", "strip"
        self.DEFAULT_additional_uniform_dict =  additional_uniform_dict.copy()
        
    def draw(self, view: glm.mat4, proj: glm.mat4, additional_uniform_dict: dict | None = None) -> None:
        #ここでのadditional_uniform_dictは，変更分だけ記述
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uView, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(self.uProj, 1, False, glm.value_ptr(proj))
        
        # 1) まず既定値を送信
        self.set_glUniform(self.DEFAULT_additional_uniform_dict)
        # 2) 次に変更分を上書き
        if additional_uniform_dict:
            self.set_glUniform(additional_uniform_dict)
        
        gl_mode = self.specify_gl_mode(self.draw_mode)
        self.geo.draw_elements(gl_mode, self.draw_mode)
        
    def set_glUniform(self, uniform_dict:dict) -> None:
        for key, value in uniform_dict.items():
            loc = getattr(self, key, None)
            if loc is None or loc == -1:
                continue #selfが見つからないならスキップ
            # 1) float
            if isinstance(value, float):
                GL.glUniform1f(loc, float(value))
                continue
            # 2) glm.vec3
            if hasattr(value, "x") and hasattr(value, "y") and hasattr(value, "z"):
                GL.glUniform3f(loc, float(value.x), float(value.y), float(value.z))
                continue
            #必要に応じて追加
        
    @staticmethod
    def specify_gl_mode(draw_mode: str) -> int:
        """set_elements で指定したタグから OpenGL のプリミティブ列挙値へ変換"""
        draw_mode = draw_mode.lower() #ぜんぶ小文字に変換
        if draw_mode in ("triangles", "triangle", "tris"):
            return GL.GL_TRIANGLES
        if draw_mode in ("strip", "triangle_strip", "tri_strip"):
            return GL.GL_TRIANGLE_STRIP
        if draw_mode in ("lines", "line"):
            return GL.GL_LINES
        if draw_mode in ("line_strip",):
            return GL.GL_LINE_STRIP
        if draw_mode in ("points", "point"):
            return GL.GL_POINTS
        # 必要に応じて追加
        raise ValueError(f"Unsupported draw mode draw_mode: {draw_mode}")

def create_nonobject_renderers(target_position:glm.vec3): #適宜追加
    # ------------Checkerboard インスタンス-------------
    # UV 正方形（0..1）。位置計算は VS 側で CK_MIN/CK_MAX を使う構成を踏襲
    checker_verts = np.array(
        [[0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]], dtype=np.float32
    )
    checker_idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

    # 追加ユニフォームは L（セル長）
    checker = NonObjectRenderer(
        name="checker",
        vertices=checker_verts,
        indices=checker_idx,
        draw_mode="tris",
        additional_uniform_dict={"L": float(param_changable["checkerboard"]["length"])},
    )

    # ------------Ray インスタンス-------------
    # ストリップ描画用のパラメトリック頂点列を生成
    slices = 32
    ray_params = np.zeros((2 * (slices + 1), 2), dtype=np.float32)
    k = 0
    for i in range(slices + 1):
        t = i / float(slices)
        ray_params[k, 0] = 0.0; ray_params[k, 1] = t; k += 1   # p0側リム
        ray_params[k, 0] = 1.0; ray_params[k, 1] = t; k += 1   # p1側リム

    ray_idx = np.arange(ray_params.shape[0], dtype=np.uint32)

    # 追加ユニフォームは端点と太さ（uP0/uP1/uR0/uR1）
    ray = NonObjectRenderer(
        name="ray",
        vertices=ray_params,
        indices=ray_idx,
        draw_mode="strip",
        additional_uniform_dict={"uP0":glm.vec3(0, 0, 0), "uP1":glm.vec3(0, 0, 0), "uR0":float(1e-4), "uR1":float(0.05)},
    )
    
    # -------------- カメラ注視点を示す青点 --------------
    blue_point_vert = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # ダミーの点　local座標
    blue_point_idx = np.array([0], dtype=np.uint32)

    cam_target_point = NonObjectRenderer(
        name="cam_target_point",
        vertices=blue_point_vert,
        indices=blue_point_idx,
        draw_mode="points",
        additional_uniform_dict={"position": target_position}  # GLSL側で加算
    )

    return checker, ray, cam_target_point