from tools import np

def compute_normals(vertices:np.ndarray, tri_indices:np.ndarray) -> np.ndarray:
        # 必要に応じて三角形法線から頂点法線を自動計算
        v = vertices
        t = tri_indices
        normals = np.zeros_like(v)
        for tri in t:
            p0, p1, p2 = v[tri]
            n = np.cross(p1 - p0, p2 - p0)
            normals[tri] += n
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        # ゼロ除算を回避
        norm[norm == 0] = 1
        normals = normals / norm
        return normals #(頂点数，3)の法線ベクトル配列