#version 330 core


// 入力は [0,1] のUV（4頂点の矩形）
layout(location=0) in vec2 in_uv;

uniform mat4 uView;
uniform mat4 uProj;
uniform float L;

vec2 CK_MIN  = vec2(-L, -L);
vec2 CK_MAX  = vec2( L,  L);

// フラグメントでチェッカー計算する用に、ワールドXYを渡す
out vec2 v_worldXY;

void main(){
    // UV→ワールドXY（XY平面 z=0）
    vec2 xy = mix(CK_MIN, CK_MAX, in_uv);
    v_worldXY = xy;

    // 位置（XY, 0）をクリップへ
    vec4 worldPos = vec4(xy, 0.0, 1.0);
    gl_Position = uProj * uView * worldPos;
}
