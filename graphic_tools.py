from tools import np, working_dir, npFloat, npInt
import ctypes
from OpenGL import GL

def compute_normals(vertices:np.ndarray, tri_indices:np.ndarray) -> np.ndarray:
        # 必要に応じて三角形法線から頂点法線を自動計算
        v = vertices
        t = tri_indices
        normals = np.zeros_like(v, dtype=np.float32)  
        for tri in t:
            p0, p1, p2 = v[tri]
            n = np.cross(p1 - p0, p2 - p0)
            normals[tri] += n
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        # ゼロ除算を回避
        norm[norm == 0] = 1
        normals = normals / norm
        return normals #(頂点数，3)の法線ベクトル配列

def build_GLProgram(vert_filename: str, frag_filename: str) -> int:
        vert_src = load_shader(vert_filename)
        frag_src = load_shader(frag_filename)
        prog = GL.glCreateProgram()
        for src, stype in [(vert_src, GL.GL_VERTEX_SHADER), (frag_src, GL.GL_FRAGMENT_SHADER)]:
            sh = GL.glCreateShader(stype)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
            if not ok:
                info = GL.glGetShaderInfoLog(sh).decode()
                raise RuntimeError(info)
            GL.glAttachShader(prog, sh)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            info = GL.glGetProgramInfoLog(prog).decode()
            raise RuntimeError(info)
        return prog
    
# --- GLSLシェーダ読込 ---
def load_shader(filename: str) -> str:
    """
    GLSLシェーダファイルを読み込み。precision行は自動削除。
    """
    filepath = working_dir / "Shader" / filename
    with open(filepath, encoding="utf-8") as f:
        code = f.read()
    code = "\n".join([l for l in code.splitlines() if not l.strip().startswith("precision")])
    return code

class GLGeometry:
    """
    VAO/VBO/EBO の薄いラッパ。頂点属性は「配列ごとに別VBO」を基本とし、
    任意の index buffer を名前付きで複数持てる（'tris','lines'等）。
    """
    def __init__(self):
        self.vao = GL.glGenVertexArrays(1)
        
        # バッファIDの辞書
        # name → バッファID（GLuint）
        #   attrN : 頂点属性VBO（位置、法線、UVなど）
        #   lines / tris : EBO（インデックスバッファ）
        self.buffers = {}    
        
        # インデックスの要素数の辞書
        self.counts  = {}    

    # ---- 追加: コンテキストマネージャ ------------
    class VAOBinder:
        def __init__(self, parent):
            self.parent = parent
        def __enter__(self):
            GL.glBindVertexArray(self.parent.vao)
        def __exit__(self, exc_type, exc_val, exc_tb):
            GL.glBindVertexArray(0)

    def bind_scope(self):
        """with構文用: VAOをbindし、終了時に自動でunbind"""
        return GLGeometry.VAOBinder(self)
    # ----------------------------------------------

    def add_array(self, loc: int, data: np.ndarray, comps: int) -> None:
        """頂点配列VBOを作り、VAOに loc でバインドする（float配列前提）。"""
        vbo = GL.glGenBuffers(1)
        self.buffers[f'attr{loc}'] = vbo
        with self.bind_scope():
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(loc)
            GL.glVertexAttribPointer(loc, comps, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

    def set_elements(self, name: str, idx: np.ndarray) -> None:
        """index buffer を登録（unsigned int 前提）。"""
        ebo = GL.glGenBuffers(1)
        self.buffers[name] = ebo
        self.counts[name]  = int(idx.size)
        with self.bind_scope():
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL.GL_STATIC_DRAW)

    def draw_elements(self, mode, name: str) -> None:
        with self.bind_scope():
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[name])
            GL.glDrawElements(mode, self.counts[name], GL.GL_UNSIGNED_INT, None)