#version 330 core
uniform vec4 uColor;
in vec3 vNormal;
in vec3 vFragPos;
out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(lightPos - vFragPos);
    vec3 V = normalize(viewPos  - vFragPos);
    vec3 R = reflect(-L, N);

    float ambient = 0.35;  // ★ここを大きめに！
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(V, R), 0.0), 32.0);

    vec3 color = uColor.rgb * (ambient + diff) + vec3(1.0) * spec * 0.3;
    FragColor = vec4(color, uColor.a);
}
