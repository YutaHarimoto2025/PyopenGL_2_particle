#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(location = 3) in vec4 iModelRow0;
layout(location = 4) in vec4 iModelRow1;
layout(location = 5) in vec4 iModelRow2;
layout(location = 6) in vec4 iModelRow3;

layout(location = 7) in vec3 iColor; // インスタンスごとの色

uniform mat4 uView, uProj;

out vec3 vNormal;
out vec3 vFragPos;
out vec3 vColor;

void main(){
    mat4 model = mat4(iModelRow0, iModelRow1, iModelRow2, iModelRow3);

    vec4 worldPos = model * vec4(aPos, 1.0);
    vFragPos = worldPos.xyz;

    mat3 normalMat = mat3(transpose(inverse(model)));
    vNormal = normalize(normalMat * aNormal);

    vColor = iColor;

    gl_Position = uProj * uView * worldPos;
}
