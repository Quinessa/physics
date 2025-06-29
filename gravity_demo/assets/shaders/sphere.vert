#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 mvp;
uniform mat4 model;

out vec3 v_normal;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_normal = mat3(transpose(inverse(model))) * in_normal;
}
