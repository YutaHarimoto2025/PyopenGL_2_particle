#version 330 core
uniform vec2 resolution;
uniform vec2 mouse;
uniform float time;
out vec4 FragColor;

void main() {
  float t = time;
  vec2 r = resolution;
  vec2 m = mouse;
  vec2 p = (gl_FragCoord.xy*2. - r) / min(r.x, r.y),
       q = (gl_FragCoord.xy*1. - r * m) / min(r.x, r.y);
  float R = 1.0 - length(q)*10.5 * length(p / sin(t*2.))*10.5,
        G = 1.0 - length(q / cos(t*3.))*7.5 * length(p)*10.5,
        B = 1.0 - length(q)*4.5 * length(p)*10.5;
  FragColor = vec4(R, G, B, 1.);
}
