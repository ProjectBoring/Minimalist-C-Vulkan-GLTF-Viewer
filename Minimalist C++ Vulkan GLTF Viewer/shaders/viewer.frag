#version 450 core

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inUv; // Add UV input from vertex shader

layout(location = 0) out vec4 outColor;

// NEW: Declare a combined image sampler uniform
// This binding must match the binding in createGraphicsPipeline and createDescriptorPoolAndSet (0 in this case).
layout(binding = 0) uniform sampler2D texSampler;

void main() {
    // Sample the texture using the interpolated UV coordinates
    vec4 textureColor = texture(texSampler, inUv);

    // Simple diffuse lighting calculation
    vec3 lightDir = normalize(vec3(0.5, 1.0, 0.0)); // Example light direction
    float diffuse = max(dot(normalize(inNormal), lightDir), 0.0);

    // Combine texture color with a basic lighting factor
    outColor = textureColor * vec4(vec3(diffuse) * 0.7 + 0.3, 1.0); // 0.3 ambient, 0.7 diffuse
    // Uncomment the line below to just see the raw texture for debugging:
    // outColor = textureColor; 
}