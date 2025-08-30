#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

#define USE_INDIVISUAL_COLOR // コメントアウトするかで切り替える
 
#ifdef USE_INDIVISUAL_COLOR
in vec3 vColor;       // 個別色入力
#else
uniform vec3 uColor;  // 統一色
#endif

uniform vec3 uLightPos;
uniform vec3 uViewPos;

out vec4 FragColor;

void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vFragPos);
    float diff = max(dot(N, L), 0.0);

    vec3 V = normalize(uViewPos - vFragPos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 8.0);

#ifdef USE_INDIVISUAL_COLOR
    vec3 baseColor = vColor;
#else
    vec3 baseColor = uColor;
#endif

    vec3 ambient  = 0.2 * baseColor;
    vec3 diffuse  = 0.7 * baseColor * diff;
    vec3 specular = 0.1 * vec3(1.0) * spec;

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
