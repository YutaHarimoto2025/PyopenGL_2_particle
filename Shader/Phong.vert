#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;   // 法線
layout(location = 2) in vec2 aTexCoord; // テクスチャUV

uniform mat4 uModel, uView, uProj;
uniform mat3 uNormalMatrix;

out vec3 vNormal;     // 法線（フラグメントシェーダへ）
out vec3 vFragPos;    // 頂点のワールド座標
out vec2 vTexCoord;   // テクスチャ座標

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;
    vNormal  = normalize(uNormalMatrix * aNormal); // 法線変換
    vTexCoord = aTexCoord;
    gl_Position = uProj * uView * worldPos;
}