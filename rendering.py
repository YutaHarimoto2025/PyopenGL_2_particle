# rendering.py
import numpy as np
from OpenGL import GL
import glm
import ctypes

from tools import param_changable, param
from graphic_tools import load_shader, build_GLProgram, GLGeometry

class Renderer:
    def __init__(self):
        self.prog = build_GLProgram(vert_filename=param.shader.vert, frag_filename=param.shader.frag)
        # ---- cache uniform locations
        GL.glUseProgram(self.prog)
        uniform_names = {
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
        self._get_uniform_loc(prog=self.prog, names=uniform_names)  
        
        #その他のset up
        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glEnable(GL.GL_CULL_FACE); #スクリーン上で反時計回りが表面，ウラ面を描画しない
        GL.glCullFace(GL.GL_BACK) # #ウラ面を描画しない　＃GL_CULL_FACEのデフォ
        GL.glFrontFace(GL.GL_CCW) # 反時計回りが表面 #GL_CULL_FACEのデフォ
        GL.glDepthMask(GL.GL_TRUE)   
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_MULTISAMPLE) #アンチエイリアス
        
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL) # ポリゴンモードを確認（塗りつぶし）
        # GL.glEnable(GL.GL_DEPTH_CLAMP) # 深度クランプを有効化（オプション）
    
        # 透明なものを描く
        # GL.glEnable(GL.GL_BLEND)
        # GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # 必要なら奥→手前の順にソート描画

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
        model = obj.model_mat
        rgba = obj.color
        GL.glUseProgram(self.prog)
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
    
    # ---- Non Object Rendering Methods ---- 
    def init_checkerboard(self):
        # シェーダファイル（中に CK_MIN/CK_MAX/CK_CELL/COLORS を const で記述）
        self.checker_prog = build_GLProgram(vert_filename="Checker.vert", frag_filename="Checker.frag")
        self._get_uniform_loc(prog=self.checker_prog, names={"uView", "uProj","L"}, prefix="checker_")

        # UV 正方形（0..1）。位置計算は VS 側で CK_MIN/CK_MAX を使って行う
        verts = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]], dtype=np.float32)
        idx   = np.array([0,1,2, 0,2,3], dtype=np.uint32)
        geo = GLGeometry()
        geo.add_array(0, verts, 2)     # layout(location=0) in vec2 in_uv;
        geo.set_elements('tris', idx)
        self.checker_geo = geo
        
    def draw_checkerboard(self, view: glm.mat4, proj: glm.mat4):
        if self.checker_prog is None or self.checker_geo is None or not param_changable["checkerboard"]["enable"]:
            return
        GL.glUseProgram(self.checker_prog)
        GL.glUniformMatrix4fv(self.checker_uView, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(self.checker_uProj, 1, False, glm.value_ptr(proj))
        GL.glUniform1f(self.checker_L, param_changable["checkerboard"]["length"])
        self.checker_geo.draw_elements(GL.GL_TRIANGLES, 'tris')
        GL.glUseProgram(self.prog) #戻す
        
    def init_ray(self):
        # シンプルな単色ライン用シェーダ（下に最小の GLSL を付けます）
        self.ray_prog = build_GLProgram(vert_filename="Ray.vert", frag_filename="Ray.frag")
        self._get_uniform_loc(prog=self.ray_prog, names={"uView", "uProj"}, prefix="ray_")

        # 2頂点のライン（座標は draw 時にその都度差し替え）
        verts = np.zeros((2, 3), dtype=np.float32)
        idx   = np.array([0, 1], dtype=np.uint32)

        geo = GLGeometry()
        geo.add_array(0, verts, 3)         # layout(location=0) in vec3 in_pos;
        geo.set_elements('lines', idx)     # インデックス: 0-1 を線分描画
        self.ray_geo = geo
    
    def draw_ray(self, view: glm.mat4, proj: glm.mat4,
             p0: glm.vec3, p1: glm.vec3,
             width: float = 10.0,
             on_top: bool = True) -> None:
        """
        p0->p1 の線分を描く
        on_top=True なら深度無効で常に手前表示。
        """
        if self.ray_prog is None or self.ray_geo is None:
            return

        GL.glUseProgram(self.ray_prog)
        GL.glUniformMatrix4fv(self.ray_uView, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(self.ray_uProj, 1, False, glm.value_ptr(proj))

        # 毎フレーム：2頂点を更新（小サイズなので add_array の再アップロードで十分）
        v = np.array([[float(p0.x), float(p0.y), float(p0.z)],
                    [float(p1.x), float(p1.y), float(p1.z)]], dtype=np.float32)
        self.ray_geo.add_array(0, v, 3)

        # 直接 OpenGL で更新（GLGeometry に VAO/VBO ハンドルがある想定の例）
        # GL.glBindVertexArray(self.ray_geo.vao)
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.ray_geo.vbos[0])
        # GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, v.nbytes, v)
        # GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        # GL.glBindVertexArray(0)
        
        # 必要なら手前固定
        if on_top:
            GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glLineWidth(width)
        self.ray_geo.draw_elements(GL.GL_LINES, 'lines')
        GL.glLineWidth(1.0)

        if on_top:
            GL.glEnable(GL.GL_DEPTH_TEST)

        # メインのプログラムに戻す（checkerboard と同じ流儀）
        GL.glUseProgram(self.prog)

        
    # ---- uniform location を安全に取得（-1なら警告）
    def _get_uniform_loc(self, prog, names:set, prefix=""):
        for name in names:
            loc = GL.glGetUniformLocation(prog, name)
            if loc == -1:
                print(f"[warn] uniform '{name}' not found (optimized out or name mismatch)")
            setattr(self, f"{prefix}{name}", loc)