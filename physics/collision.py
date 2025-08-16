from object3d import Object3D
import glm
import math

def collision_time_back(obj1: Object3D, obj2: Object3D)-> float:
    p = obj1.position - obj2.position # 相対位置ベクトル
    v = obj1.velocity - obj2.velocity # 相対速度ベクトル
    R = obj1.radius   + obj2.radius # 半径の和
    
    # 二次方程式の係数
    a = glm.dot(v, v)
    b = -2.0 * glm.dot(p, v)
    c = glm.dot(p, p) - R * R
    
    if a == 0.0:  # 相対速度がゼロなら衝突しない
        print("2つの球の相対速度が0")
        return None
    disc = b * b - 4.0 * a * c  # 判別式
    if disc < 0.0:
        print("2つの球は衝突しない")
        return None
        
    sqrt_disc = math.sqrt(disc)
    t1 = (b - sqrt_disc) / (2.0 * a)
    t2 = (b + sqrt_disc) / (2.0 * a)
    print(f"t1: {t1}, t2: {t2}")

    # 正の最小値（現時刻から巻き戻す時間）
    t_candidates = [t for t in (t1, t2) if t >= 0.0]
    return min(t_candidates) if t_candidates else None
