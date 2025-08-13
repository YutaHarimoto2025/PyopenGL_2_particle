from OpenGL import GL
import glm
from typing import Optional, Tuple, Any
import math

from tools import xp, np, xpFloat, xpInt, npFloat, npInt, rngnp, param_changable, param
from graphic_tools import compute_normals  
from object3d import Object3D

# create_obj.py
# -------------------------------------------------------
# 3Dオブジェクト生成用ファクトリ関数
# -------------------------------------------------------
def get_oneball_vertices_faces(subdiv=2, radius=0.5)-> Tuple[list, list]:
    """
    subdiv: 細分割回数 分割一回で面が4倍になる
    radius: 球半径
    戻り値: (頂点座標Nx3, 面インデックスMx3, 法線Nx3)
    """
    t = (1.0 + math.sqrt(5.0))/2.0
    verts = [(-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
             ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
             ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1)]
    faces = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
             (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
             (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
             (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    verts = [glm.normalize(glm.vec3(v))*radius for v in verts]

    def midpoint(a, b):
        m = (glm.vec3(a) + glm.vec3(b)) * 0.5
        return glm.normalize(m) * radius

    for _ in range(subdiv):
        new_faces = []
        mid_cache = {}
        for i, j, k in faces:
            def get_mid(v1, v2):
                key = tuple(sorted((v1, v2)))
                if key not in mid_cache:
                    mid_cache[key] = len(verts)
                    verts.append(midpoint(verts[v1], verts[v2]))
                return mid_cache[key]
            a = get_mid(i, j)
            b = get_mid(j, k)
            c = get_mid(k, i)
            new_faces += [(i, a, c), (j, b, a), (k, c, b), (a, b, c)]
        faces = new_faces

    # numpy配列に
    vertices = npFloat(verts)
    tri_indices = npInt(faces)
    return vertices, tri_indices

def create_balls(
    num: int,
    radius: float,
    posi_limit = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
    subdiv: int = 2,
) -> list:
    """
    num_ball: 生成する球の個数
    subdiv: 細分割回数
    radius: 球の半径
    spacing: 各球の中心間距離（簡易的にx軸上に並べる例）
    """
    obj_list = []
    for i in range(num):
        # 球のメッシュ生成
        vertices, tri_indices = get_oneball_vertices_faces(subdiv=subdiv, radius=radius)

        # 配置例：x軸上に等間隔で並べる
        posi = tuple(
            rngnp.uniform(low=lim[0], high=lim[1]) for lim in posi_limit
        )

        obj = Object3D(
            vertices=vertices,
            tri_indices=tri_indices,
            color=(1.0, 1.0, 1.0), 
            name=f"ball",
            radius=radius,
            name_posi_local=(0,0,0),
            posi=posi,
            is_move=True
        )
        obj_list.append(obj)
    return obj_list

def one_ball(radius: float = 0.8, posi: Tuple|None=None, color: Tuple|None=None, texture_path = None) ->list:
    vertices, tri_indices = get_oneball_vertices_faces(subdiv=3, radius=radius)
    obj = Object3D(
        vertices=vertices,
        tri_indices=tri_indices,
        uv_mode = "spherical",
        color=color if color is not None else (1,1,1), #(0.2, 0.2, 0.8), # 青
        name="ball",
        radius=radius,
        name_posi_local=(0, 0, 0),
        posi= posi if posi is not None else (0, 0, radius),

        is_move=True,
        texture_path=texture_path
    )
    return [obj]

def create_boxes(scale=(1, 1, 1)) -> list:
    """
    一辺1の直方体。各面ごとに頂点を独立させ、UV(0..1)を割り当てる。
    戻りは Object3D のリスト（既存API踏襲）。
    """
    obj_list = []

    # 6面 × 4頂点 = 24頂点（各面に独立UVを持たせる）
    V = npFloat([
        # -Z face
        [-0.5,-0.5,-0.5], [ 0.5,-0.5,-0.5], [ 0.5, 0.5,-0.5], [-0.5, 0.5,-0.5],
        # +Z face
        [-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5], [ 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        # -X face
        [-0.5,-0.5,-0.5], [-0.5, 0.5,-0.5], [-0.5, 0.5, 0.5], [-0.5,-0.5, 0.5],
        # +X face
        [ 0.5,-0.5,-0.5], [ 0.5, 0.5,-0.5], [ 0.5, 0.5, 0.5], [ 0.5,-0.5, 0.5],
        # -Y face
        [-0.5,-0.5,-0.5], [ 0.5,-0.5,-0.5], [ 0.5,-0.5, 0.5], [-0.5,-0.5, 0.5],
        # +Y face
        [-0.5, 0.5,-0.5], [ 0.5, 0.5,-0.5], [ 0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
    ])

    # 各面の4頂点に [0,0],[1,0],[1,1],[0,1] を割り当て（面ごと独立）
    UV_face = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    UV = np.vstack([UV_face for _ in range(6)]).astype(np.float32)

    # 各面 2トライアングル（0,1,2, 0,2,3）× 6面
    TRI = []
    for f in range(6):
        o = 4 * f  # オフセット
        
        if f == 0:      # -Z face: 外から見て時計回り → 反時計回りに修正
            TRI += [o+0, o+3, o+2,  o+0, o+2, o+1]
        elif f == 1:    # +Z face: 外から見て反時計回り
            TRI += [o+0, o+1, o+2,  o+0, o+2, o+3]
        elif f == 2:    # -X face: 外から見て時計回り → 反時計回りに修正  
            TRI += [o+0, o+3, o+2,  o+0, o+2, o+1]
        elif f == 3:    # +X face: 外から見て反時計回り
            TRI += [o+0, o+1, o+2,  o+0, o+2, o+3]
        elif f == 4:    # -Y face: 外から見て反時計回り
            TRI += [o+0, o+1, o+2,  o+0, o+2, o+3]
        elif f == 5:    # +Y face: 外から見て時計回り → 反時計回りに修正
            TRI += [o+0, o+3, o+2,  o+0, o+2, o+1]

    TRI = npInt(TRI)

    # ライン（ワイヤーフレーム）。面ごとに4辺（重複OKで簡易）
    LINES = []
    face_edges = [(0,1),(1,2),(2,3),(3,0)]
    for f in range(6):
        o = 4 * f
        for a,b in face_edges:
            LINES += [o+a, o+b]
    LINES = npInt(LINES)

    color = (1.0, 1.0, 0.2)  # 黄色

    box = Object3D(
        vertices=V,
        line_indices=LINES,
        tri_indices=TRI,
        # uvs=UV, #指定しなくても自動生成される
        posi=(0.5, 0.5, 0.5),
        color=color,
        scale=scale,
        name="box",
        name_posi_local=(0,0,0),
        # texture_path=param_changable["box_texture"],
    )
    obj_list.append(box)
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
