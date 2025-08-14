import ctypes
from OpenGL import GL
from PIL import Image
import math
from OpenGL.GLU import gluUnProject
import glm

from tools import np, working_dir, npFloat, npInt

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
        shaders = []
        for src, stype in [(vert_src, GL.GL_VERTEX_SHADER), (frag_src, GL.GL_FRAGMENT_SHADER)]:
            sh = GL.glCreateShader(stype)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            ok = GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS)
            if not ok:
                info = GL.glGetShaderInfoLog(sh).decode()
                raise RuntimeError(info)
            GL.glAttachShader(prog, sh)
            shaders.append(sh)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            info = GL.glGetProgramInfoLog(prog).decode()
            raise RuntimeError(info)
        
        # シェーダオブジェクトの切り離しと破棄
        for sh in shaders:
            GL.glDetachShader(prog, sh)
            GL.glDeleteShader(sh)
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

def load_texture(figpath):
    img = Image.open(figpath).convert("RGBA")  # 透過ありでもOK
    img_data = img.tobytes()
    width, height = img.size

    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    # ミップマップ＆フィルタ
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)

    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                 GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data) #画像をGPU目盛りにコピー
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return tex

# --- マウスイベントに使うレイキャストまわり ---
def _ray_hit_plane(ro: glm.vec3,
                   rd: glm.vec3,
                   plane_point: glm.vec3 = glm.vec3(0, 0, 0),
                   plane_normal: glm.vec3 = glm.vec3(0, 0, 1))-> glm.vec3 | None:
    """
    レイと任意平面（デフォでz=0）の交点を返す。
    ro: レイ原点 (Ray Origin)
    rd: レイ方向 (Ray Direction, 正規化ずみ)
    plane_point: 平面上の任意の点
    plane_normal: 平面の法線ベクトル（正規化推奨）
    戻り値: 交点 glm.vec3 または None（交差しない）
    """
    denom = glm.dot(plane_normal, rd)
    if abs(denom) < 1e-9:  # レイが平面と平行
        print("Ray is parallel to the plane, so does not hit.")
        return None

    t = glm.dot(plane_point - ro, plane_normal) / denom
    if t <= 0.0:
        print("the plane is behind the ray half straight line, so does not hit.")
        return None

    return ro + t * rd

def _ray_hit_sphere(ro: glm.vec3, rd: glm.vec3, center: glm.vec3, radius: float):
    """レイと球の交差。最小の正の t を返す。ヒット無しは None"""
    oc = ro - center
    b = glm.dot(oc, rd)          # 注意: a=1（rd正規化前提）
    c = glm.dot(oc, oc) - radius * radius
    disc = b*b - c               # 判別式（a=1, 2b→b*2 を省略）
    if disc < 0.0:
        return None
    sqrt_d = math.sqrt(disc)
    t1 = -b - sqrt_d
    t2 = -b + sqrt_d
    # 最小の正の解を採用
    if t1 > 1e-6: return t1
    if t2 > 1e-6: return t2
    return None


def seam_split(positions, normals, uvs, faces):
    """説明文
    U方向の継ぎ目を分割して、UVが[0,1]範囲に収まるようにする。"""
    faces = faces.reshape(-1, 3)
    
    P = positions.tolist()
    N = normals.tolist() if normals is not None and len(normals) else None
    UV = uvs.tolist()
    new_faces = []
    for (i,j,k) in faces:
        us = [UV[i][0], UV[j][0], UV[k][0]]
        if max(us) - min(us) > 0.5:
            tri = []
            for idx in (i,j,k):
                u,v = UV[idx]
                if u < 0.5:
                    P.append(positions[idx].tolist())
                    if N is not None: N.append(normals[idx].tolist())
                    UV.append([u+1.0, v])
                    tri.append(len(P)-1)
                else:
                    tri.append(idx)
            new_faces.append(tri)
        else:
            new_faces.append([i,j,k])
    return np.array(P, np.float32), (np.array(N, np.float32) if N is not None else None), \
           np.array(UV, np.float32), np.array(new_faces, np.uint32)

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
            
# ---- UV 計算関数群 -------------
def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _bbox(vertices: np.ndarray):
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    size = np.maximum(vmax - vmin, 1e-6)  # ゼロ割回避
    return vmin, vmax, size

def _uv_spherical(vertices: np.ndarray,
                  center: np.ndarray | None = None,
                  up_axis: str = "z",
                  rot_deg: float = 0.0) -> np.ndarray:
    if center is None:
        center = vertices.mean(axis=0)
    p = vertices - center
    n = _normalize_rows(p)
    # 経線基準の回転（任意）
    rot = math.radians(rot_deg) #初期角

    if up_axis == "z":
        theta = np.arctan2(n[:, 1], n[:, 0]) + rot   # (y,x)
        lat   = np.arcsin(n[:, 2])                   # z が上
    elif up_axis == "x":
        theta = np.arctan2(n[:, 2], n[:, 1]) + rot   # (z,y)
        lat   = np.arcsin(n[:, 0])                   # x が上
    else:  # "y" デフォルト
        theta = np.arctan2(n[:, 2], n[:, 0]) + rot   # (z,x)
        lat   = np.arcsin(n[:, 1])                   # y が上

    u = (theta + math.pi) / (2.0 * math.pi)          # 0..1
    v = 0.5 - (lat / math.pi)                        # 0..1（北=0, 南=1）
    return np.stack([u, v], axis=1).astype(np.float32)

def _uv_cylindrical(vertices: np.ndarray, axis: int | None = None) -> np.ndarray:
    # 最長軸を高さにする（axis=Noneなら自動）
    vmin, _, size = _bbox(vertices)
    if axis is None:
        axis = int(np.argmax(size))  # 0=x,1=y,2=z
    p = vertices
    if axis == 1:   # Y軸が高さ
        theta = np.arctan2(p[:, 2], p[:, 0])
        u = (theta + math.pi) / (2.0 * math.pi)
        v = (p[:, 1] - vmin[1]) / size[1]
    elif axis == 0: # X軸が高さ
        theta = np.arctan2(p[:, 2], p[:, 1])
        u = (theta + math.pi) / (2.0 * math.pi)
        v = (p[:, 0] - vmin[0]) / size[0]
    else:           # Z軸が高さ
        theta = np.arctan2(p[:, 1], p[:, 0])
        u = (theta + math.pi) / (2.0 * math.pi)
        v = (p[:, 2] - vmin[2]) / size[2]
    return np.stack([u, v], axis=1).astype(np.float32)

def _uv_box(vertices: np.ndarray, normals: np.ndarray | None = None) -> np.ndarray:
    """
    トリプラナーに近い簡易Box投影（各頂点の支配法線成分で面を選ぶ）。
    伸縮はモデルのAABBに基づき[0,1]へ正規化。
    """
    vmin, _, size = _bbox(vertices)
    if normals is None or len(normals) != len(vertices):
        # 法線が無い場合は中心からの方向で代用
        normals = _normalize_rows(vertices - vertices.mean(axis=0))

    ax = np.argmax(np.abs(normals), axis=1)  # 0=x面,1=y面,2=z面
    u = np.empty(len(vertices), dtype=np.float32)
    v = np.empty(len(vertices), dtype=np.float32)

    # X面（±X）：(u,v)=(-z,y)
    m = (ax == 0)
    if np.any(m):
        u[m] = (-(vertices[m, 2]) - vmin[2]) / size[2]
        v[m] = ( (vertices[m, 1]) - vmin[1]) / size[1]

    # Y面（±Y）：(u,v)=(x,-z)
    m = (ax == 1)
    if np.any(m):
        u[m] = ( (vertices[m, 0]) - vmin[0]) / size[0]
        v[m] = (-(vertices[m, 2]) - vmin[2]) / size[2]

    # Z面（±Z）：(u,v)=(x,y)
    m = (ax == 2)
    if np.any(m):
        u[m] = ( (vertices[m, 0]) - vmin[0]) / size[0]
        v[m] = ( (vertices[m, 1]) - vmin[1]) / size[1]

    # 範囲を[0,1]に
    u = np.mod(u, 1.0)
    v = np.mod(v, 1.0)
    return np.stack([u, v], axis=1).astype(np.float32)

def compute_uvs(
    vertices: np.ndarray,
    tri_indices: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    mode: str = "auto",
    spherical_threshold: float = 0.06,
) -> np.ndarray:
    """
    共通UV自動生成
    - mode: "auto" | "box" | "spherical" | "cylindrical"
    - spherical_threshold: AUTO時に球判定する距離分散しきい値（小さいほど球と判断）
    """
    v = np.asarray(vertices, dtype=np.float32)

    if mode == "box":
        return _uv_box(v, normals)
    if mode == "spherical":
        return _uv_spherical(v)
    if mode == "cylindrical":
        return _uv_cylindrical(v)

    # --- AUTO ヒューリスティック ---
    # 形状の“球らしさ”= 中心からの距離の相対標準偏差で判定
    center = v.mean(axis=0)
    r = np.linalg.norm(v - center, axis=1)
    rel_std = (r.std() / (r.mean() + 1e-12)) if len(r) else 1.0

    if rel_std < spherical_threshold:
        return _uv_spherical(v)
    else:
        return _uv_box(v, normals)
