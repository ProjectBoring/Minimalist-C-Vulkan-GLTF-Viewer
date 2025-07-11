#include "ProjectionMatrix.h"
#include "Main.h"

glm::mat4 getProjectionMatrix() {
    float aspect = static_cast<float>(WIN_W) / static_cast<float>(WIN_H);
    glm::mat4 proj = glm::perspective(glm::radians(45.0f), aspect, 0.01f, 100.0f);
    proj[1][1] *= -1;  // Vulkan Y coordinate fix
    return proj;
}
