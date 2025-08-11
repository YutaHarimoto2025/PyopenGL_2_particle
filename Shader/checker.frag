#version 330 core

// ===== チェッカーパラメータ（固定パラメータ）=====
const float CK_CELL  = 1.0;
const vec3  CK_COLOR1 = vec3(0.92, 0.92, 0.92);
const vec3  CK_COLOR2 = vec3(0.08, 0.08, 0.08);

in vec2 v_worldXY;
out vec4 fragColor;

void main(){
    vec2 uv = floor(v_worldXY / CK_CELL);
    float m = mod(uv.x + uv.y, 2.0);
    vec3 col = mix(CK_COLOR1, CK_COLOR2, m);
    fragColor = vec4(col, 1.0);
}
