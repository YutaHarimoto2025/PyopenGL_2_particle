#version 330 core

layout(location=0) in vec2 aParam;   // x: endpoint(0 or 1), y: t in [0,1] for angle

uniform mat4 uView, uProj;
uniform vec3 uP0, uP1;   // 円錐の両端（ワールド）
uniform float uR0, uR1;  // p0/p1 側の半径

void orthonormalBasis(in vec3 axis, out vec3 U, out vec3 V){
    vec3 t = (abs(axis.y) < 0.99) ? vec3(0,1,0) : vec3(1,0,0);
    U = normalize(cross(axis, t));
    V = normalize(cross(axis, U));
}

void main(){
    float endpoint = aParam.x;       // 0.0 -> p0側, 1.0 -> p1側
    float t = aParam.y;              // 0..1 → 角度
    float ang = t * 6.28318530718;   // 2π

    vec3 axis = normalize(uP1 - uP0);
    vec3 U, V; orthonormalBasis(axis, U, V);

    float r   = mix(uR0, uR1, endpoint);
    vec3  C   = mix(uP0, uP1, endpoint);      // 中心
    vec3  rim = C + (U*cos(ang) + V*sin(ang)) * r;

    gl_Position = uProj * uView * vec4(rim, 1.0);
}
