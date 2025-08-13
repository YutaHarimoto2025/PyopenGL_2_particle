#version 330 core
layout(location = 0) in vec3 aPos; //頂点位置 local座標
uniform vec3 position; // world座標
uniform mat4 uView, uProj;

void main() {
    gl_Position = uProj * uView * vec4(aPos + position, 1.0);
    gl_PointSize = 5.0; // 必要に応じて大きさ指定 正方形の辺をpixel数で指定
}