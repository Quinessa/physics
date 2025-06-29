#version 330 core
in vec3 v_normal;

uniform vec3 color;

out vec4 fragColor;

void main() {
    vec3 n = normalize(v_normal);
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diff = max(dot(n, lightDir), 0.0);
    vec3 diffuse = color * diff;
    vec3 ambient = color * 0.2;
    fragColor = vec4(pow(diffuse + ambient, vec3(1.0/2.2)), 1.0);
}
