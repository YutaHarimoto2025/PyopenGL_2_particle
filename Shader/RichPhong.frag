#version 330 core
in vec3 vNormal;
in vec3 vFragPos;

out vec4 FragColor;

uniform vec3  uLightPos;
uniform vec3  uViewPos;
uniform vec4  uColor;          // ← vec4( rgb, a )
uniform int   uShadingMode;    // 0:Phong, 1:Toon, 2:Phong+Rim
uniform float uRimStrength;    // 0〜1
uniform float uGamma;          // 例: 2.2

vec3 phong(vec3 N, vec3 L, vec3 V){
    float diff = max(dot(N, L), 0.0);
    vec3  R    = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), 32.0);
    float ambient = 0.2;
    // 色は uColor.rgb を使う
    return uColor.rgb * (ambient + diff) + vec3(1.0) * spec * 0.25;
}

vec3 toon(vec3 N, vec3 L){
    float d = max(dot(N,L),0.0);
    float k = d < 0.25 ? 0.1 : (d < 0.5 ? 0.4 : (d < 0.8 ? 0.7 : 1.0));
    return uColor.rgb * (0.15 + 0.85*k);
}

void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vFragPos);
    vec3 V = normalize(uViewPos  - vFragPos);

    vec3 col;
    if(uShadingMode == 1){
        col = toon(N, L);
    }else{
        col = phong(N, L, V);
        if(uShadingMode == 2){
            float rim = 1.0 - max(dot(N, V), 0.0);//視線にたいして面が横向きほど輪郭ぽい
            vec3 rimColor = vec3(1.0, 1.0, 1.0);
            col += rimColor *rim * uRimStrength; // リムライト
        }
    }

    // ガンマ補正（線形→sRGB）
    col = pow(max(col, 0.0), vec3(1.0 / uGamma));
    FragColor = vec4(col, uColor.a);  // ← アルファを反映
}
