#version 330

uniform mat4 u_projMatrix;
in vec4 Position;

void main() {
    vec4 pos = u_projMatrix * Position;
    pos.z += 0.01;
    gl_Position = pos;
}
