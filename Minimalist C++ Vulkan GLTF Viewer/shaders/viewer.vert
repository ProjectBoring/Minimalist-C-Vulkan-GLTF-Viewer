#version 450 core

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv; // Add UV input

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUv; // Add UV output to fragment shader

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model; // Add model matrix for normal transformation
} pushConstants;

void main() {
    gl_Position = pushConstants.mvp * vec4(inPos, 1.0);
    // Transform normal by the model matrix (ignoring scaling for simplicity here)
    // For correct normal transformation with non-uniform scaling, you'd use inverse transpose of model matrix.
    outNormal = normalize(mat3(pushConstants.model) * inNormal); 
    outUv = inUv; // Pass UVs to fragment shader
}