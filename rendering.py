# rendering.py
import numpy as np
from OpenGL import GL
import glm

from tools import param_changable

class Renderer:
    def __init__(self, vert_src: str, frag_src: str):
        # ---- compile & link
        self.prog = GL.glCreateProgram()
        for src, stype in [(vert_src, GL.GL_VERTEX_SHADER), (frag_src, GL.GL_FRAGMENT_SHADER)]:
            sh = GL.glCreateShader(stype)
            GL.glShaderSource(sh, src); GL.glCompileShader(sh)
            ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
            if not ok: raise RuntimeError(GL.glGetShaderInfoLog(sh).decode())
            GL.glAttachShader(self.prog, sh)
        GL.glLinkProgram(self.prog)
        if not GL.glGetProgramiv(self.prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(self.prog).decode())

        # ---- cache uniform locations
        GL.glUseProgram(self.prog)
        # ---- uniform location を安全に取得（-1なら警告）
        def _loc(name):
            loc = GL.glGetUniformLocation(self.prog, name)
            if loc == -1:
                print(f"[warn] uniform '{name}' not found (optimized out or name mismatch)")
            return loc
        self.uModel   = _loc("uModel")
        self.uView    = _loc("uView")
        self.uProj    = _loc("uProj")
        self.uViewPos = _loc("uViewPos")
        self.uLightPos= _loc("uLightPos")
        self.uColor   = _loc("uColor")
        self.uShading = _loc("uShadingMode")
        self.uRim     = _loc("uRimStrength")
        self.uGamma   = _loc("uGamma")
        self.uNormalM = _loc("uNormalMatrix")  
        
        #その他のset up
        GL.glEnable(GL.GL_DEPTH_TEST); GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glEnable(GL.GL_CULL_FACE); GL.glCullFace(GL.GL_BACK) #閉じたメッシュならON
        GL.glEnable(GL.GL_BLEND)

    def set_common(self, cam_posi, view, proj):
        light_pos = (5.0, 5.0, 5.0)
        shading_mode=2 # 0:Phong / 1:Toon / 2:Phong+Rim
        rim=0.2 #0~1 1に近いと輪郭が白くなる
        gamma=1.4 #1で補正なし，<1で暗くなる， >1で明るくなる
        # --- ビュー・プロジェクション行列の生成 ---
        GL.glUseProgram(self.prog)
        GL.glUniformMatrix4fv(self.uView, 1, False, glm.value_ptr(view))
        GL.glUniformMatrix4fv(self.uProj, 1, False, glm.value_ptr(proj))
        GL.glUniform3f(self.uViewPos, *cam_posi)
        GL.glUniform3f(self.uLightPos, *light_pos)
        GL.glUniform1i(self.uShading, shading_mode)
        GL.glUniform1f(self.uRim, float(rim))
        GL.glUniform1f(self.uGamma, float(gamma))

    def set_model_and_normal(self, model: glm.mat4):
        # uModel
        GL.glUniformMatrix4fv(self.uModel, 1, False, glm.value_ptr(model))
        # uNormalMatrix (mat3) —— 非一様スケール対応
        nmat = glm.mat3(glm.transpose(glm.inverse(glm.mat3(model))))
        GL.glUniformMatrix3fv(self.uNormalM, 1, False, glm.value_ptr(nmat))

    def set_color(self, rgba):
        GL.glUniform4f(self.uColor, *rgba)
        