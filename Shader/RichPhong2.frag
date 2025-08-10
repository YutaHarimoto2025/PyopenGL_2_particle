#version 330 core
in vec3 vNormal;
in vec3 vFragPos;
in vec2 v_uv;

out vec4 FragColor;

// === 既存 ===
uniform vec3  uLightPos;
uniform vec3  uViewPos;
uniform vec4  uColor;          // base color (rgb,a)
uniform int   uShadingMode;    // 0:Phong, 1:Toon, 2:Phong+Rim
uniform float uRimStrength;    // 0..1
uniform float uGamma;          // 例: 2.2（>0 のときsRGBエンコード）

// === 追加: マテリアル/ライティング調整 ===
uniform vec3  uLightColor   = vec3(1.0); // ライトの色（白）
uniform float uAmbient      = 0.20;  // 環境光
uniform float uSpecularStr  = 0.25;  // 鏡面強度
uniform float uShininess    = 32.0;  // ハイライト鋭さ
uniform float uDiffuse      = 1.0;  // 拡散反射の強さ

// === 追加: リムライト調整（色/カーブ） ===
uniform vec3  uRimColor     = vec3(1.0);
uniform float uRimExponent  = 1.0;   // リムの立ち上がり（1=線形, 大きいほどエッジ寄り）

// === 追加: UVベース（テクスチャ/チェッカー） ===
uniform int   uUseTexture   = 0;     // 0:使わない, 1:使う
uniform sampler2D uTex; 
uniform int   uTexIsSRGB    = 1;     // 1: テクスチャはsRGB → 線形に直す
uniform int   uUseUVChecker = 0;     // 0:使わない, 1:使う
uniform float uUVCell       = 1.0;   // チェッカーのセル幅（UV空間）
uniform vec3  uUVColor1     = vec3(0.92);
uniform vec3  uUVColor2     = vec3(0.08);

vec3 phong(vec3 base, vec3 N, vec3 L, vec3 V){
    // 拡散反射
    float diff = max(dot(N, L), 0.0);
    vec3 diffuse  = uDiffuse * diff * base * uLightColor;

    // 環境光
    vec3 ambient  = uAmbient * base * uLightColor;

    // 鏡面反射（Blinn-Phong）
    vec3  H       = normalize(L + V);
    float specVal = pow(max(dot(N, H), 0.0), uShininess);
    vec3 specular = uSpecularStr * specVal * uLightColor;

    // 合成
    return ambient + diffuse + specular;
}

vec3 toon(vec3 base, vec3 N, vec3 L){
    float d = max(dot(N, L), 0.0);
    float k = d < 0.25 ? 0.1 : (d < 0.5 ? 0.4 : (d < 0.8 ? 0.7 : 1.0));
    return base * (0.15 + 0.85 * k);
}

vec3 makeBaseColor(){
    // uColor をベースに、テクスチャ or UVチェッカーを合成
    vec3 base = uColor.rgb;

    if(uUseTexture == 1){
        vec4 tex = texture(uTex, v_uv);
        vec3 albedo = tex.rgb;
        if(uTexIsSRGB == 1){
            // sRGB→線形  (uGamma=2.2想定。別に uTexGamma を作ってもOK)
            albedo = pow(albedo, vec3(uGamma));
        }
        base *= albedo;
        // アルファ合成するなら FragColor.a に tex.a を掛ける（末尾で反映）
    } else if(uUseUVChecker == 1){
        vec2 uv = floor(v_uv / uUVCell);
        float m = mod(uv.x + uv.y, 2.0);
        vec3 c  = mix(uUVColor1, uUVColor2, m);
        base = c; // 乗算にしたければ: base *= c;
    }
    return base;
}

void main(){
    vec3 N = normalize(vNormal);
    // if (!gl_FrontFacing) N = -N;  // 裏面は法線反転 透けてみえる
    if (!gl_FrontFacing) {
        discard;  // 裏面のピクセルを描画しない
    }

    vec3 L = normalize(uLightPos - vFragPos);
    vec3 V = normalize(uViewPos  - vFragPos);

    vec3 base = makeBaseColor();

    vec3 col;
    if(uShadingMode == 1){
        col = toon(base, N, L);
    }else{
        col = phong(base, N, L, V);
        if(uShadingMode == 2){
            // リムライト（エッジを強調）。smoothに立ち上げる
            float rim = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), uRimExponent);
            col += uRimColor * rim * uRimStrength;
        }
    }

    // 線形→sRGB 変換（フレームバッファが sRGB 有効なら不要）
    if(uGamma > 0.0){
        col = pow(max(col, 0.0), vec3(1.0 / uGamma));
    }

    // テクスチャアルファを使う場合はここで掛ける：
    // float alpha = uColor.a * (uUseTexture == 1 ? texture(uAlbedoTex, v_uv).a : 1.0);
    float alpha = uColor.a;

    FragColor = vec4(col, alpha);
}
