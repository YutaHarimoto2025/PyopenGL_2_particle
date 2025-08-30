#version 330 core
layout(location = 0) in vec3 aPos;    // 単位球の頂点 (原点中心)

uniform mat4 uModel, uView, uProj;

out vec3 vNormal;
out vec3 vFragPos;

void main() {
    // ワールド座標
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;

    // 球の法線 = 頂点位置ベクトル (単位球ベースなので normalize)
    vNormal = normalize(mat3(uModel) * aPos);

    gl_Position = uProj * uView * worldPos;
}
