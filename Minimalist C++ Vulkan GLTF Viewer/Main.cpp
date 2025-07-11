// ============================================================================
// Full Vulkan + GLFW + fastgltf viewer
// - Handles window resizing.
// - Reads all nodes/meshes from a glTF file.
// - Renders with a vertex/fragment shader pair.
// - Model transformation (orbit, zoom) is done on the GPU via push constants.
// - Uses a device-local vertex buffer for optimal performance.
// ============================================================================

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Use fastgltf for glTF loading
#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/base64.hpp>

#define GLFW_INCLUDE_VULKAN
#define NOMINMAX // Prevent Windows headers from defining min/max macros
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#ifdef _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <filesystem> // Required for fastgltf file loading
#include <cstdint>    // Required for uint types
#include <iomanip>    // For std::setw, std::setfill
#include <curl/curl.h>

// Include generated SPIR-V shader headers (required)
#include "shaders/viewer_vert_spv.h"
#include "shaders/viewer_frag_spv.h"
#include "resource.h"
#include "Main.h" // For WIN_W and WIN_H and other window settings

// ---------- Configuration ----------------------------------------------------
const char* GLTF_PATH = "C:/Users/admin/Downloads/MyAvatar.gltf"; // ❗ CHANGE TO YOUR MODEL
const char* GLTF_URL = "https://raw.githubusercontent.com/SaschaWillems/Vulkan-glTF-PBR/refs/heads/master/data/models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf"; // ❗ CHANGE TO YOUR MODEL
/*
MORE URIS:
GLTF RESOURCE is supported inside resource.h!
Just add a gltf file to Minimalist C++ Vulkan GLTF Viewer.rc and then call loadGLTFFromResource(int resourceId)

Example entry in gltf header area:
IDR_GLTF1               gltf                    "models\\Shedletsky.gltf"
*/

// ---------- Helper Functions -----------------------------------------------
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename);
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// Helper function to read file into a vector of uint8_t for fastgltf
static std::vector<std::uint8_t> readFile_u8(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<std::uint8_t> buffer(fileSize);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

// Download a URL into a byte vector using libcurl
static size_t curlWrite(void* contents, size_t size, size_t nmemb, void* userp) {
    auto* vec = static_cast<std::vector<std::uint8_t>*>(userp);
    size_t total = size * nmemb;
    vec->insert(vec->end(), (std::uint8_t*)contents, (std::uint8_t*)contents + total);
    return total;
}

std::vector<std::uint8_t> downloadUrl(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl)
        throw std::runtime_error("curl_easy_init failed");

    std::vector<std::uint8_t> data;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::string err = curl_easy_strerror(res);
        curl_easy_cleanup(curl);
        throw std::runtime_error("Failed to download '" + url + "': " + err);
    }

    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);
    if (httpCode != 200) {
        throw std::runtime_error(
            "Request to '" + url + "' returned HTTP status " + std::to_string(httpCode));
    }
    return data;
}

// ---------- Vertex & Matrix Structs ----------------------------------------
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct PushConstants {
    glm::mat4 mvp;
    glm::mat4 model;
};

// Represents a loaded texture on the GPU
// VulkanTexture struct to hold image, memory, and view
struct VulkanTexture {
    VkImage image;
    VkDeviceMemory imageMemory;
    VkImageView imageView;
};

// Represents a glTF material
// Material struct to hold material properties
struct Material {
    glm::vec4 baseColorFactor{ 1.0f, 1.0f, 1.0f, 1.0f };
    std::optional<uint32_t> baseColorTextureIndex; // Index into our gTextures array
};

// A primitive is a single draw call with a single material
// Primitive struct to link geometry to materials
struct Primitive {
    uint32_t firstIndex;
    uint32_t indexCount;
    int materialIndex; // Index into gMaterials
};

// A mesh is a collection of primitives
// Mesh struct to hold primitives
struct Mesh {
    std::vector<Primitive> primitives;
};


// ---------- Globals ----------------------------------------------------------
// GLFW & Camera
GLFWwindow* gWindow{};
bool gFramebufferResized = false;
// Model orientation (changed via mouse dragging)
float gYaw   = glm::radians(DEFAULT_MODEL_YAW);
float gPitch = glm::radians(DEFAULT_MODEL_PITCH);
float gZoom  = DEFAULT_ZOOM;
// Camera orientation (rotate the camera around the origin)
float gCamYaw   = glm::radians(DEFAULT_CAMERA_YAW);
float gCamPitch = glm::radians(DEFAULT_CAMERA_PITCH);
bool gMouseDown = false;
double gLastX = 0, gLastY = 0;

// Vulkan Core
VkInstance gInst{};
VkSurfaceKHR gSurface{};
VkPhysicalDevice gGpu = VK_NULL_HANDLE;
VkDevice gDev{};
uint32_t gGfxQueueFamily{};
VkQueue gGfxQueue{}, gPresentQueue{};

// Swapchain & related resources
VkSwapchainKHR gSwapchain{};
VkFormat gSwapchainFormat{};
VkExtent2D gSwapchainExtent{};
std::vector<VkImage> gSwapchainImages;
std::vector<VkImageView> gSwapchainImageViews;
std::vector<VkFramebuffer> gFramebuffers;

// Depth Buffer
VkImage gDepthImage{};
VkDeviceMemory gDepthImageMemory{};
VkImageView gDepthImageView{};

// Pipeline
VkRenderPass gRenderPass{};
VkPipelineLayout gPipelineLayout{};
VkPipeline gGraphicsPipeline{};

// Buffers & Data
std::vector<Vertex> gVertices;
VkBuffer gVertexBuffer{};
VkDeviceMemory gVertexBufferMemory{};
// Index buffer globals
VkBuffer gIndexBuffer{};
VkDeviceMemory gIndexBufferMemory{};

std::vector<uint32_t> gIndices; // Indices data
std::vector<Mesh> gMeshes;      // Meshes from GLTF
std::vector<Material> gMaterials; // Materials from GLTF
std::vector<VulkanTexture> gTextures; // Loaded Vulkan textures

// Descriptors for Textures
VkDescriptorSetLayout gDescriptorSetLayout{};
VkDescriptorPool gDescriptorPool{};          
std::vector<VkDescriptorSet> gMaterialDescriptorSets; // One descriptor set per textured material
VkSampler gTextureSampler{};                  // Global sampler

// Commands & Sync
VkCommandPool gCommandPool{};
std::vector<VkCommandBuffer> gCommandBuffers;
VkSemaphore gImageAvailableSemaphore{};
VkSemaphore gRenderFinishedSemaphore{};
VkFence gRenderFence{};

// ---------- Forward Declarations -------------------------------------------
void initVulkan();
void mainLoop();
void cleanup();
void cleanupSwapchain();
void recreateSwapchain();
void createInstance();
void createSurface();
void pickPhysicalDevice();
void createLogicalDevice();
void createSwapchain();
void createImageViews();
void createRenderPass();
VkShaderModule createShaderModule(const unsigned char* codeData, size_t codeSize);
void createDepthResources();
void createGraphicsPipeline();
void createFramebuffers();
void createCommandPool();
void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory); // signature
void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size); // signature
void createVertexBuffer();
void createIndexBuffer();
void createCommandBuffers();
void createSyncObjects();
void drawFrame();
uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
// Texture related helper declarations
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
VkImageView createImageView(VkImage image, VkFormat format);
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
void createTextureSampler();
void createDescriptorPoolAndSet();
// Updated processNode signature
void processNode(const fastgltf::Asset& asset, size_t nodeIndex, Mesh& outMesh, std::vector<uint32_t>& indices, std::vector<Vertex>& vertices);
// Helper to convert a parsed fastgltf::Asset into our runtime structures
void processAsset(const fastgltf::Asset& asset, const std::filesystem::path& baseDir);
void loadGLTF(const char* path);
std::vector<std::uint8_t> downloadUrl(const std::string& url);
void loadGLTFBuffer(const std::vector<std::uint8_t>& data, const std::filesystem::path& baseDir = {});
void loadGLTFFromMemory(const uint8_t* data, size_t size, const std::filesystem::path& baseDir = {});
void loadGLTFFromURL(const char* url);
#ifdef _WIN32
void loadGLTFFromResource(int resourceId);
#endif

// ============================================================================
//                                   MAIN
// ============================================================================
int main() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    gWindow = glfwCreateWindow(WIN_W, WIN_H, "Vulkan glTF Viewer", nullptr, nullptr);

    glfwSetFramebufferSizeCallback(gWindow, [](GLFWwindow*, int, int) { gFramebufferResized = true; });

    glfwSetMouseButtonCallback(gWindow, [](GLFWwindow* window, int button, int action, int) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            gMouseDown = (action == GLFW_PRESS);
            glfwGetCursorPos(window, &gLastX, &gLastY);
        }
        });

    glfwSetCursorPosCallback(gWindow, [](GLFWwindow*, double xpos, double ypos) {
        if (!gMouseDown) return;
        float dx = static_cast<float>(xpos - gLastX);
        float dy = static_cast<float>(gLastY - ypos);
        gLastX = xpos;
        gLastY = ypos;
        gYaw += dx * 0.005f;
        gPitch = std::clamp(gPitch + dy * 0.005f, -1.55334f, 1.55334f);
        });

    glfwSetScrollCallback(gWindow, [](GLFWwindow*, double, double yoffset) {
        gZoom = std::clamp(gZoom * (yoffset > 0 ? 0.9f : 1.1f), 0.5f, 10.f);
        });

    try {
        // STEP 1: Initialize CORE Vulkan components (Instance, Device, Sampler, Swapchain, RenderPass, PipelineLayout)
        // These do NOT depend on GLTF data for their creation.
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createTextureSampler(); // Sampler can be created now
        createSwapchain();
        createImageViews();
        createRenderPass();
        createDepthResources();
        createGraphicsPipeline(); // Pipeline layout (with descriptor set layout) is created here
        createFramebuffers();
        createCommandPool();
        createCommandBuffers(); // These are independent of model data
        createSyncObjects();    // These are independent of model data

        // STEP 2: Load GLTF data
        // This will now successfully create Vulkan images for textures because gDev is valid.
        //loadGLTF(GLTF_PATH);
        //loadGLTFFromURL(GLTF_URL);
#ifdef _WIN32
        loadGLTFFromResource(IDR_GLTF1); // pretty much loads from memory
#endif

        // STEP 3: Create Vulkan resources that DEPEND on GLTF data
        // These functions now have the data they need.
        createVertexBuffer();       // Uses gVertices
        createIndexBuffer();        // Uses gIndices
        createDescriptorPoolAndSet(); // Uses gTextures, gMaterials

        mainLoop();
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    cleanup();
    return EXIT_SUCCESS;
}

void initVulkan() {
    createInstance();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createTextureSampler(); // Create sampler before pipeline
    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createGraphicsPipeline(); // Graphics pipeline now expects descriptor set layout
    createFramebuffers();
    createCommandPool();
    createVertexBuffer(); // These use gVertices/gIndices populated by loadGLTF
    createIndexBuffer();  // Index buffer
    createDescriptorPoolAndSet(); // Allocate and update descriptor sets
    createCommandBuffers();
    createSyncObjects();
}

void mainLoop() {
    while (!glfwWindowShouldClose(gWindow)) {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(gDev);
}

void loadGLTFFromMemory(const uint8_t* data, size_t size, const std::filesystem::path& baseDir) {
    std::vector<std::uint8_t> vec(data, data + size);
    loadGLTFBuffer(vec, baseDir);
}

void loadGLTFFromURL(const char* url) {
    std::vector<std::uint8_t> vec = downloadUrl(url);
    loadGLTFBuffer(vec, {});
}

#ifdef _WIN32
void loadGLTFFromResource(int resourceId) {
    HMODULE module = GetModuleHandleW(nullptr);
    HRSRC res = FindResourceW(module, MAKEINTRESOURCEW(resourceId), L"gltf");
    if (!res)
        throw std::runtime_error("FindResource failed");
    HGLOBAL handle = LoadResource(module, res);
    if (!handle)
        throw std::runtime_error("LoadResource failed");
    DWORD size = SizeofResource(module, res);
    const void* ptr = LockResource(handle);
    if (!ptr)
        throw std::runtime_error("LockResource failed");
    loadGLTFFromMemory(static_cast<const uint8_t*>(ptr), static_cast<size_t>(size));
}
#endif

void loadGLTF(const char* path) {
    auto filePath = std::filesystem::path(path);
    fastgltf::Parser parser;

    auto bufferResult = fastgltf::GltfDataBuffer::FromPath(filePath);
    if (bufferResult.error() != fastgltf::Error::None) {
        throw std::runtime_error(std::string("Failed to open glTF file: ") + std::string(fastgltf::getErrorMessage(bufferResult.error())));
    }
    fastgltf::GltfDataBuffer data = std::move(bufferResult.get());

    fastgltf::Expected<fastgltf::Asset> assetResult = parser.loadGltf(
        data, filePath.parent_path(),
        fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);

    if (auto error = assetResult.error(); error != fastgltf::Error::None) {
        throw std::runtime_error(std::string("Failed to load glTF: ") + std::string(fastgltf::getErrorMessage(error)));
    }
    fastgltf::Asset& asset = assetResult.get();

    processAsset(asset, filePath.parent_path());
}

void cleanupSwapchain() {
    vkDestroyImageView(gDev, gDepthImageView, nullptr);
    vkDestroyImage(gDev, gDepthImage, nullptr);
    vkFreeMemory(gDev, gDepthImageMemory, nullptr);

    for (auto framebuffer : gFramebuffers) {
        vkDestroyFramebuffer(gDev, framebuffer, nullptr);
    }

    vkDestroyPipeline(gDev, gGraphicsPipeline, nullptr);
    vkDestroyPipelineLayout(gDev, gPipelineLayout, nullptr);
    vkDestroyRenderPass(gDev, gRenderPass, nullptr);

    for (auto imageView : gSwapchainImageViews) {
        vkDestroyImageView(gDev, imageView, nullptr);
    }

    vkDestroySwapchainKHR(gDev, gSwapchain, nullptr);
}

void cleanup() {
    cleanupSwapchain();

    // Cleanup for texture resources
    vkDestroySampler(gDev, gTextureSampler, nullptr);
    vkDestroyDescriptorPool(gDev, gDescriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(gDev, gDescriptorSetLayout, nullptr);

    // Destroy textures
    for (const auto& tex : gTextures) {
        if (tex.imageView != VK_NULL_HANDLE) vkDestroyImageView(gDev, tex.imageView, nullptr);
        if (tex.image != VK_NULL_HANDLE) vkDestroyImage(gDev, tex.image, nullptr);
        if (tex.imageMemory != VK_NULL_HANDLE) vkFreeMemory(gDev, tex.imageMemory, nullptr);
    }

    vkDestroyBuffer(gDev, gIndexBuffer, nullptr);
    vkFreeMemory(gDev, gIndexBufferMemory, nullptr);
    vkDestroyBuffer(gDev, gVertexBuffer, nullptr);
    vkFreeMemory(gDev, gVertexBufferMemory, nullptr);

    vkDestroySemaphore(gDev, gRenderFinishedSemaphore, nullptr);
    vkDestroySemaphore(gDev, gImageAvailableSemaphore, nullptr);
    vkDestroyFence(gDev, gRenderFence, nullptr);

    vkDestroyCommandPool(gDev, gCommandPool, nullptr);

    vkDestroyDevice(gDev, nullptr);
    vkDestroySurfaceKHR(gInst, gSurface, nullptr);
    vkDestroyInstance(gInst, nullptr);

    glfwDestroyWindow(gWindow);
    glfwTerminate();
}

void recreateSwapchain() {
    // Handle minimization
    int width = 0, height = 0;
    glfwGetFramebufferSize(gWindow, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(gWindow, &width, &height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(gDev);

    cleanupSwapchain();

    createSwapchain();
    createImageViews();
    createRenderPass();
    createDepthResources();
    createGraphicsPipeline();
    createFramebuffers();
    // No need to recreate descriptor pool/sets or sampler unless textures change.
    // However, if the images themselves or the number of materials change,
    // you would need to recreate these. For simplicity, we're assuming static textures.
}

// ============================================================================
//                     VULKAN SETUP IMPLEMENTATION
// ============================================================================

// ... (createInstance, createSurface, pickPhysicalDevice, createLogicalDevice unchanged) ...
void createInstance() {
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "glTF Viewer";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    VkInstanceCreateInfo createInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;

    if (vkCreateInstance(&createInfo, nullptr, &gInst) != VK_SUCCESS) {
        throw std::runtime_error("failed to create instance!");
    }
}

void createSurface() {
    if (glfwCreateWindowSurface(gInst, gWindow, nullptr, &gSurface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void pickPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(gInst, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(gInst, &deviceCount, devices.data());
    gGpu = devices[0];
}

void createLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gGpu, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(gGpu, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(gGpu, i, gSurface, &presentSupport);
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT && presentSupport) {
            gGfxQueueFamily = i;
            break;
        }
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueCreateInfo.queueFamilyIndex = gGfxQueueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo createInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());

    if (vkCreateDevice(gGpu, &createInfo, nullptr, &gDev) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }
    vkGetDeviceQueue(gDev, gGfxQueueFamily, 0, &gGfxQueue);
    gPresentQueue = gGfxQueue;
}


void createSwapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gGpu, gSurface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(gGpu, gSurface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(gGpu, gSurface, &formatCount, formats.data());

    VkSurfaceFormatKHR surfaceFormat = formats[0];
    for (const auto& availableFormat : formats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            surfaceFormat = availableFormat;
            break;
        }
    }

    gSwapchainFormat = surfaceFormat.format;

    if (capabilities.currentExtent.width != UINT32_MAX) {
        gSwapchainExtent = capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(gWindow, &width, &height);
        gSwapchainExtent = {
            std::clamp((uint32_t)width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp((uint32_t)height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface = gSurface;
    createInfo.minImageCount = capabilities.minImageCount > 0 ? capabilities.minImageCount : 3;
    createInfo.imageFormat = gSwapchainFormat;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = gSwapchainExtent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = VK_PRESENT_MODE_FIFO_KHR; // V-Sync
    createInfo.clipped = VK_TRUE;

    if (vkCreateSwapchainKHR(gDev, &createInfo, nullptr, &gSwapchain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    uint32_t imageCount;
    vkGetSwapchainImagesKHR(gDev, gSwapchain, &imageCount, nullptr);
    gSwapchainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(gDev, gSwapchain, &imageCount, gSwapchainImages.data());
}

void createImageViews() {
    gSwapchainImageViews.resize(gSwapchainImages.size());
    for (size_t i = 0; i < gSwapchainImages.size(); i++) {
        VkImageViewCreateInfo createInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        createInfo.image = gSwapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = gSwapchainFormat;
        createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;
        if (vkCreateImageView(gDev, &createInfo, nullptr, &gSwapchainImageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

void createDepthResources() {
    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;

    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = gSwapchainExtent.width;
    imageInfo.extent.height = gSwapchainExtent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(gDev, &imageInfo, nullptr, &gDepthImage) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(gDev, gDepthImage, &memRequirements);

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(gDev, &allocInfo, nullptr, &gDepthImageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate depth image memory!");
    }
    vkBindImageMemory(gDev, gDepthImage, gDepthImageMemory, 0);

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = gDepthImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthFormat;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    if (vkCreateImageView(gDev, &viewInfo, nullptr, &gDepthImageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create depth image view!");
    }
}

void createRenderPass() {
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = gSwapchainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorAttachmentRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef{ 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::vector<VkAttachmentDescription> attachments = { colorAttachment, depthAttachment };
    VkRenderPassCreateInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(gDev, &renderPassInfo, nullptr, &gRenderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

VkShaderModule createShaderModule(const unsigned char* codeData, size_t codeSize) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(codeData);
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(gDev, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}

void createGraphicsPipeline() {
    // Use the global embedded shader data and sizes generated by your Python script
    // Ensure you have #include "viewer_vert_spv.h" and #include "viewer_frag_spv.h" at the top of your file
    VkShaderModule vertShaderModule = createShaderModule(g_viewer_vert_spv, g_viewer_vert_spv_size);
    VkShaderModule fragShaderModule = createShaderModule(g_viewer_frag_spv, g_viewer_frag_spv_size);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);
    attributeDescriptions[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) };
    attributeDescriptions[1] = { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) };
    attributeDescriptions[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{ 0.0f, 0.0f, (float)gSwapchainExtent.width, (float)gSwapchainExtent.height, 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, gSwapchainExtent };
    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0, 1, &viewport, 1, &scissor };

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE; // Current setting, based on your previous fix
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE; // Current setting, based on your previous fix

    VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlending{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    VkDescriptorSetLayoutBinding samplerLayoutBinding{};
    samplerLayoutBinding.binding = 0;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &samplerLayoutBinding;

    if (vkCreateDescriptorSetLayout(gDev, &layoutInfo, nullptr, &gDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);

    std::vector<VkDescriptorSetLayout> descriptorSetLayouts = { gDescriptorSetLayout };
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(gDev, &pipelineLayoutInfo, nullptr, &gPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.layout = gPipelineLayout;
    pipelineInfo.renderPass = gRenderPass;
    pipelineInfo.subpass = 0;

    if (vkCreateGraphicsPipelines(gDev, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &gGraphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(gDev, fragShaderModule, nullptr);
    vkDestroyShaderModule(gDev, vertShaderModule, nullptr);
}

void createFramebuffers() {
    gFramebuffers.resize(gSwapchainImageViews.size());
    for (size_t i = 0; i < gSwapchainImageViews.size(); i++) {
        std::vector<VkImageView> attachments = { gSwapchainImageViews[i], gDepthImageView };
        VkFramebufferCreateInfo framebufferInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        framebufferInfo.renderPass = gRenderPass;
        framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = gSwapchainExtent.width;
        framebufferInfo.height = gSwapchainExtent.height;
        framebufferInfo.layers = 1;
        if (vkCreateFramebuffer(gDev, &framebufferInfo, nullptr, &gFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void createCommandPool() {
    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = gGfxQueueFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(gDev, &poolInfo, nullptr, &gCommandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(gDev, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(gDev, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(gDev, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }
    vkBindBufferMemory(gDev, buffer, bufferMemory, 0);
}

void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = gCommandPool;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(gDev, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(gGfxQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(gGfxQueue);

    vkFreeCommandBuffers(gDev, gCommandPool, 1, &commandBuffer);
}

void createVertexBuffer() {
    if (gVertices.empty()) return;
    VkDeviceSize bufferSize = sizeof(gVertices[0]) * gVertices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(gDev, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, gVertices.data(), (size_t)bufferSize);
    vkUnmapMemory(gDev, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gVertexBuffer, gVertexBufferMemory);

    copyBuffer(stagingBuffer, gVertexBuffer, bufferSize);

    vkDestroyBuffer(gDev, stagingBuffer, nullptr);
    vkFreeMemory(gDev, stagingBufferMemory, nullptr);
}

// createIndexBuffer implementation
void createIndexBuffer() {
    if (gIndices.empty()) return;
    VkDeviceSize bufferSize = sizeof(gIndices[0]) * gIndices.size();

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(gDev, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, gIndices.data(), (size_t)bufferSize);
    vkUnmapMemory(gDev, stagingBufferMemory);

    createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, gIndexBuffer, gIndexBufferMemory);

    copyBuffer(stagingBuffer, gIndexBuffer, bufferSize);

    vkDestroyBuffer(gDev, stagingBuffer, nullptr);
    vkFreeMemory(gDev, stagingBufferMemory, nullptr);
}

void createCommandBuffers() {
    gCommandBuffers.resize(gFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = gCommandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)gCommandBuffers.size();
    if (vkAllocateCommandBuffers(gDev, &allocInfo, gCommandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void createSyncObjects() {
    VkSemaphoreCreateInfo semaphoreInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    if (vkCreateSemaphore(gDev, &semaphoreInfo, nullptr, &gImageAvailableSemaphore) != VK_SUCCESS ||
        vkCreateSemaphore(gDev, &semaphoreInfo, nullptr, &gRenderFinishedSemaphore) != VK_SUCCESS ||
        vkCreateFence(gDev, &fenceInfo, nullptr, &gRenderFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to create synchronization objects for a frame!");
    }
}

void drawFrame() {
    vkWaitForFences(gDev, 1, &gRenderFence, VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(gDev, gSwapchain, UINT64_MAX, gImageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vkResetFences(gDev, 1, &gRenderFence);
    vkResetCommandBuffer(gCommandBuffers[imageIndex], 0);

    // Begin recording
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(gCommandBuffers[imageIndex], &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    std::vector<VkClearValue> clearValues(2);
    clearValues[0].color = { {0.1f, 0.1f, 0.15f, 1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };
    VkRenderPassBeginInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    renderPassInfo.renderPass = gRenderPass;
    renderPassInfo.framebuffer = gFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = gSwapchainExtent;
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();
    vkCmdBeginRenderPass(gCommandBuffers[imageIndex], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(gCommandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, gGraphicsPipeline);

    VkBuffer vertexBuffers[] = { gVertexBuffer };
    VkDeviceSize offsets[] = { 0 };
    vkCmdBindVertexBuffers(gCommandBuffers[imageIndex], 0, 1, vertexBuffers, offsets);
    vkCmdBindIndexBuffer(gCommandBuffers[imageIndex], gIndexBuffer, 0, VK_INDEX_TYPE_UINT32);

    PushConstants pushConsts;
    // Calculate the camera position based on the configurable yaw/pitch values
    glm::vec3 camPos = glm::vec3(0.f, 0.f, 3.f);
    glm::mat4 camRot = glm::rotate(glm::mat4(1.0f), gCamYaw, glm::vec3(0.f, 1.f, 0.f));
    camRot = glm::rotate(camRot, gCamPitch, glm::vec3(1.f, 0.f, 0.f));
    camPos = glm::vec3(camRot * glm::vec4(camPos, 1.0f));
    // View matrix uses the rotated camera position so the scene can start at
    // a developer-defined angle.
    glm::mat4 view = glm::lookAt(camPos, glm::vec3(0.f), glm::vec3(0.f, 1.f, 0.f));
    glm::mat4 proj = glm::perspective(glm::radians(45.f), gSwapchainExtent.width / (float)gSwapchainExtent.height, 0.1f, 100.f);
    proj[1][1] *= -1; // Invert Y for Vulkan

    pushConsts.model = glm::mat4(1.0f);
    pushConsts.model = glm::rotate(pushConsts.model, gYaw, glm::vec3(0.f, 1.f, 0.f));
    pushConsts.model = glm::rotate(pushConsts.model, gPitch, glm::vec3(1.f, 0.f, 0.f));
    pushConsts.model = glm::scale(pushConsts.model, glm::vec3(1.f / gZoom));
    pushConsts.mvp = proj * view * pushConsts.model;
    vkCmdPushConstants(gCommandBuffers[imageIndex], gPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushConsts);

    // Iterate through meshes and primitives, binding the correct descriptor set for each
    for (const auto& mesh : gMeshes) {
        for (const auto& primitive : mesh.primitives) {
            // Check if the primitive has a valid material index and if that material has a descriptor set
            if (primitive.materialIndex != -1 && primitive.materialIndex < gMaterialDescriptorSets.size()) {
                VkDescriptorSet materialDescriptorSet = gMaterialDescriptorSets[primitive.materialIndex];
                if (materialDescriptorSet != VK_NULL_HANDLE) {
                    // Bind the descriptor set specific to this primitive's material
                    vkCmdBindDescriptorSets(gCommandBuffers[imageIndex], VK_PIPELINE_BIND_POINT_GRAPHICS, gPipelineLayout, 0, 1, &materialDescriptorSet, 0, nullptr);
                }
                else {
                    // Fallback or warning: Material exists but has no valid texture/descriptor set assigned.
                    // If no specific texture is bound for this primitive, it will inherit whatever was
                    // bound previously or use default values if nothing was bound at all.
                    // For models where some parts are textured and some are not, you might need a
                    // default "no texture" descriptor set here.
                }
            }
            else {
                // Primitive has no material, or an invalid material index.
                // It will be rendered without a specific material texture.
                // Consider binding a default "white" or "fallback" texture if this occurs for visual consistency.
            }
            vkCmdDrawIndexed(gCommandBuffers[imageIndex], primitive.indexCount, 1, primitive.firstIndex, 0, 0);
        }
    }

    vkCmdEndRenderPass(gCommandBuffers[imageIndex]);
    if (vkEndCommandBuffer(gCommandBuffers[imageIndex]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    VkSemaphore waitSemaphores[] = { gImageAvailableSemaphore };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &gCommandBuffers[imageIndex];
    VkSemaphore signalSemaphores[] = { gRenderFinishedSemaphore };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (vkQueueSubmit(gGfxQueue, 1, &submitInfo, gRenderFence) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    VkSwapchainKHR swapChains[] = { gSwapchain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(gPresentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || gFramebufferResized) {
        gFramebufferResized = false;
        recreateSwapchain();
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }
}

// ============================================================================
//                          HELPER IMPLEMENTATIONS
// ============================================================================

uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(gGpu, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

// createImage helper
void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(gDev, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(gDev, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(gDev, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(gDev, image, imageMemory, 0);
}

// createImageView helper
VkImageView createImageView(VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(gDev, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture image view!");
    }

    return imageView;
}

// beginSingleTimeCommands helper
VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = gCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(gDev, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);
    return commandBuffer;
}

// endSingleTimeCommands helper
void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(gGfxQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(gGfxQueue); // Wait for the copy to finish

    vkFreeCommandBuffers(gDev, gCommandPool, 1, &commandBuffer);
}

// transitionImageLayout helper
void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
}

// copyBufferToImage helper
void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { width, height, 1 };

    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

// createTextureSampler implementation
void createTextureSampler() {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE; // For simplicity
    samplerInfo.maxAnisotropy = 1.0f; // Ignored if anisotropyEnable is false
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR; // No mipmapping for simplicity
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 0.0f;

    if (vkCreateSampler(gDev, &samplerInfo, nullptr, &gTextureSampler) != VK_SUCCESS) {
        throw std::runtime_error("failed to create texture sampler!");
    }
}

// createDescriptorPoolAndSet implementation
void createDescriptorPoolAndSet() {
    // Define the pool sizes. We need enough for each material that might have a texture.
    // Each descriptor set will contain one combined image sampler.
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = 0; // Initialize to 0, count actual textured materials

    // Count how many materials actually have a texture to size the pool correctly
    for (const auto& mat : gMaterials) {
        if (mat.baseColorTextureIndex.has_value() && mat.baseColorTextureIndex.value() < gTextures.size() && gTextures[mat.baseColorTextureIndex.value()].imageView != VK_NULL_HANDLE) {
            poolSize.descriptorCount++;
        }
    }

    if (poolSize.descriptorCount == 0) {
        std::cout << "No textured materials found, skipping descriptor pool and set creation.\n";
        return; // No textures to bind
    }

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = poolSize.descriptorCount; // We will allocate one set for each textured material

    if (vkCreateDescriptorPool(gDev, &poolInfo, nullptr, &gDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }

    // Allocate and update a descriptor set for EACH textured material
    gMaterialDescriptorSets.resize(gMaterials.size(), VK_NULL_HANDLE); // Initialize with VK_NULL_HANDLE

    for (size_t i = 0; i < gMaterials.size(); ++i) {
        const auto& mat = gMaterials[i];
        if (mat.baseColorTextureIndex.has_value() && mat.baseColorTextureIndex.value() < gTextures.size() && gTextures[mat.baseColorTextureIndex.value()].imageView != VK_NULL_HANDLE) {
            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool = gDescriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &gDescriptorSetLayout; // Use the single layout we created

            if (vkAllocateDescriptorSets(gDev, &allocInfo, &gMaterialDescriptorSets[i]) != VK_SUCCESS) {
                // Handle allocation failure, maybe continue to next material or throw
                std::cerr << "Warning: Failed to allocate descriptor set for material " << i << ". Skipping.\n";
                gMaterialDescriptorSets[i] = VK_NULL_HANDLE; // Ensure it's null if allocation failed
                continue;
            }

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = gTextures[mat.baseColorTextureIndex.value()].imageView;
            imageInfo.sampler = gTextureSampler;

            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = gMaterialDescriptorSets[i]; // Bind to THIS material's set
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite.descriptorCount = 1;
            descriptorWrite.pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(gDev, 1, &descriptorWrite, 0, nullptr);
            std::cout << "Descriptor set updated for material " << i << ", using image index: " << mat.baseColorTextureIndex.value() << "\n";
        }
        else {
            // Material has no texture or invalid texture index, leave its descriptor set as VK_NULL_HANDLE
            std::cout << "Material " << i << " has no valid base color texture. No descriptor set bound.\n";
        }
    }
}
// Updated processNode to populate Mesh and Primitive structures, and use gIndices
void processNode(const fastgltf::Asset& asset, size_t nodeIndex, Mesh& outMesh, std::vector<uint32_t>& indices, std::vector<Vertex>& vertices) {
    const auto& node = asset.nodes[nodeIndex];

    if (node.meshIndex.has_value()) {
        const auto& mesh = asset.meshes[*node.meshIndex];
        outMesh.primitives.reserve(mesh.primitives.size());

        for (const auto& primitive : mesh.primitives) {
            Primitive newPrimitive;
            newPrimitive.firstIndex = static_cast<uint32_t>(indices.size());
            newPrimitive.materialIndex = primitive.materialIndex.value_or(-1);

            uint32_t vertexStart = static_cast<uint32_t>(vertices.size()); // Offset for current primitive's vertices

            const fastgltf::Accessor* posAccessor = nullptr;
            if (auto it = primitive.findAttribute("POSITION"); it != primitive.attributes.end()) {
                posAccessor = &asset.accessors[it->accessorIndex];
            }
            else {
                continue; // Skip primitives without positions
            }

            const fastgltf::Accessor* normAccessor = nullptr;
            if (auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end()) {
                normAccessor = &asset.accessors[it->accessorIndex];
            }

            const fastgltf::Accessor* uvAccessor = nullptr;
            if (auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end()) {
                uvAccessor = &asset.accessors[it->accessorIndex];
            }

            std::vector<Vertex> primVertices(posAccessor->count);

            size_t i = 0;
            fastgltf::iterateAccessor<glm::vec3>(asset, *posAccessor, [&](glm::vec3 pos) { primVertices[i++].pos = pos; });

            if (normAccessor) {
                i = 0;
                fastgltf::iterateAccessor<glm::vec3>(asset, *normAccessor, [&](glm::vec3 norm) { primVertices[i++].normal = norm; });
            }
            else {
                // If no normals, set to default (e.g., zero vector or some placeholder)
                for (auto& v : primVertices) v.normal = glm::vec3(0.0f);
            }

            if (uvAccessor) {
                i = 0;
                fastgltf::iterateAccessor<glm::vec2>(asset, *uvAccessor, [&](glm::vec2 uv) { primVertices[i++].uv = uv; });
            }
            else {
                // If no UVs, set to default (e.g., zero vector)
                for (auto& v : primVertices) v.uv = glm::vec2(0.0f);
            }


            // Load Indices
            if (primitive.indicesAccessor.has_value()) {
                const auto& indexAccessor = asset.accessors[*primitive.indicesAccessor];
                newPrimitive.indexCount = static_cast<uint32_t>(indexAccessor.count);
                fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor,
                    [&](std::uint32_t index) { indices.push_back(vertexStart + index); });
            }
            else {
                // If no index buffer, assume non-indexed draw, but we still need to generate indices
                // that correspond to the order of vertices inserted.
                newPrimitive.indexCount = static_cast<uint32_t>(posAccessor->count);
                for (uint32_t j = 0; j < posAccessor->count; ++j) {
                    indices.push_back(vertexStart + j);
                }
            }

            vertices.insert(vertices.end(), primVertices.begin(), primVertices.end());
            outMesh.primitives.push_back(newPrimitive);
        }
    }

    for (auto childNodeIndex : node.children) {
        processNode(asset, childNodeIndex, outMesh, indices, vertices); // Pass through outMesh, indices, vertices
    }
}


void loadGLTFBuffer(const std::vector<std::uint8_t>& fileData, const std::filesystem::path& baseDir) {
    fastgltf::Parser parser;

    auto bufferResult = fastgltf::GltfDataBuffer::FromBytes(
        reinterpret_cast<const std::byte*>(fileData.data()), fileData.size());

    if (bufferResult.error() != fastgltf::Error::None) {
        throw std::runtime_error(std::string("Failed to create GltfDataBuffer: ") + std::string(fastgltf::getErrorMessage(bufferResult.error())));
    }
    fastgltf::GltfDataBuffer data = std::move(bufferResult.get());

    bool isBinary = fileData.size() >= 4 &&
        std::memcmp(fileData.data(), "glTF", 4) == 0;

    fastgltf::Options opts = fastgltf::Options::None;
    if (!baseDir.empty()) {
        opts |= fastgltf::Options::LoadExternalBuffers;
        opts |= fastgltf::Options::LoadExternalImages;
    }

    fastgltf::Expected<fastgltf::Asset> assetResult = isBinary
        ? parser.loadGltfBinary(data, baseDir, opts)
        : parser.loadGltfJson(data, baseDir, opts);

    if (auto error = assetResult.error(); error != fastgltf::Error::None) {
        throw std::runtime_error(std::string("Failed to load glTF: ") + std::string(fastgltf::getErrorMessage(error)));
    }
    fastgltf::Asset& asset = assetResult.get();

    processAsset(asset, baseDir);
}

void processAsset(const fastgltf::Asset& asset, const std::filesystem::path& baseDir) {
    // --- Print Asset Info ---
    std::cout << "========================================\n";
    std::cout << "glTF Asset Info (fastgltf)\n";
    std::cout << "========================================\n";
    if (asset.assetInfo.has_value()) {
        std::cout << "Asset Version: " << asset.assetInfo->gltfVersion << "\n";
        std::cout << "Asset Generator: " << asset.assetInfo->generator << "\n";
    }
    std::cout << "Images: " << asset.images.size() << "\n";
    std::cout << "========================================\n\n";

    // --- Load Textures (create Vulkan resources) ---
    std::cout << "--- Loading Textures ---\n";
    gTextures.resize(asset.images.size());
    for (size_t i = 0; i < asset.images.size(); ++i) {
        const auto& image = asset.images[i];
        std::cout << "  Image [" << i << "]: " << image.name;

        unsigned char* pixels = nullptr;
        int width = 0, height = 0, channels = 0; // channels will be 4 after stbi_load (RGBA)

        std::visit([&](auto&& source) {
            using T = std::decay_t<decltype(source)>;
            if constexpr (std::is_same_v<T, fastgltf::sources::Array>) {
                pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(source.bytes.data()), static_cast<int>(source.bytes.size()), &width, &height, &channels, 4);
            }
            else if constexpr (std::is_same_v<T, fastgltf::sources::BufferView>) {
                auto& bufferView = asset.bufferViews[source.bufferViewIndex];
                auto& buffer = asset.buffers[bufferView.bufferIndex];
                if (auto* vector_data = std::get_if<fastgltf::sources::Vector>(&buffer.data)) {
                    pixels = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(vector_data->bytes.data() + bufferView.byteOffset), static_cast<int>(bufferView.byteLength), &width, &height, &channels, 4);
                }
            }
            else if constexpr (std::is_same_v<T, fastgltf::sources::URI>) {
                auto imagePath = (baseDir / source.uri.path()).string();
                pixels = stbi_load(imagePath.c_str(), &width, &height, &channels, 4);
            }
            }, image.data);

        if (pixels) {
            std::cout << " (Decoded OK), Size: " << width << "x" << height << "\n";
            VkDeviceSize imageSize = width * height * 4; // Always 4 channels (RGBA)

            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

            void* mappedData;
            vkMapMemory(gDev, stagingBufferMemory, 0, imageSize, 0, &mappedData);
            memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
            vkUnmapMemory(gDev, stagingBufferMemory);
            stbi_image_free(pixels); // Free CPU-side pixels after copying to staging buffer

            VulkanTexture& tex = gTextures[i];
            // Create the actual GPU image
            createImage(static_cast<uint32_t>(width), static_cast<uint32_t>(height), VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, tex.image, tex.imageMemory);

            // Transition layout, copy data, transition again
            transitionImageLayout(tex.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            copyBufferToImage(stagingBuffer, tex.image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
            transitionImageLayout(tex.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            // Create image view
            tex.imageView = createImageView(tex.image, VK_FORMAT_R8G8B8A8_SRGB);

            // Clean up staging buffer
            vkDestroyBuffer(gDev, stagingBuffer, nullptr);
            vkFreeMemory(gDev, stagingBufferMemory, nullptr);
        }
        else {
            std::cout << " (Failed to decode or Empty Source)\n";
            // Important: Mark this texture as invalid or skip to prevent issues later
            gTextures[i].image = VK_NULL_HANDLE;
            gTextures[i].imageMemory = VK_NULL_HANDLE;
            gTextures[i].imageView = VK_NULL_HANDLE;
        }
    }
    std::cout << "-------------------------\n\n";

    // --- Load Materials ---
    // Populate gMaterials vector
    gMaterials.resize(asset.materials.size());
    for (size_t i = 0; i < asset.materials.size(); ++i) {
        const auto& mat = asset.materials[i];
        if (mat.pbrData.baseColorTexture.has_value()) {
            auto textureIndex = mat.pbrData.baseColorTexture->textureIndex;
            // The textureIndex points to asset.textures, which then has an imageIndex
            auto imageIndex = asset.textures[textureIndex].imageIndex;
            if (imageIndex.has_value()) {
                gMaterials[i].baseColorTextureIndex = static_cast<uint32_t>(imageIndex.value());
            }
        }
        // glTF baseColorFactor is a vec4 (RGBA)
        gMaterials[i].baseColorFactor = glm::make_vec4(mat.pbrData.baseColorFactor.data());
    }

    // --- Load Meshes and Geometry ---
    // Populate gMeshes, gVertices, gIndices
    gMeshes.resize(asset.meshes.size());
    gVertices.clear();
    gIndices.clear();

    const auto* scene = asset.defaultScene.has_value() ? &asset.scenes[*asset.defaultScene] : &asset.scenes.front();
    for (size_t nodeIndex : scene->nodeIndices) {
        // We pass a single mesh reference, and append all primitives from all nodes into it.
        // This is a simplification; a full viewer would have a scene graph.
        processNode(asset, nodeIndex, gMeshes[0], gIndices, gVertices); // Using gMeshes[0] as a catch-all
    }
    // A more correct approach for multiple meshes would be:
    // for (size_t i = 0; i < asset.meshes.size(); ++i) {
    //     processNode(asset, i, gMeshes[i], gIndices, gVertices);
    // }

    if (gVertices.empty()) {
        throw std::runtime_error("No vertices loaded from glTF file.");
    }

    // --- Normalize Model ---
    glm::vec3 min_v(std::numeric_limits<float>::max()), max_v(std::numeric_limits<float>::lowest());
    for (const auto& v : gVertices) {
        min_v = glm::min(min_v, v.pos);
    }
    for (const auto& v : gVertices) {
        max_v = glm::max(max_v, v.pos);
    }

    glm::vec3 center = (min_v + max_v) * 0.5f;
    float radius = 0.f;
    for (const auto& v : gVertices) {
        radius = glm::max(radius, glm::length(v.pos - center));
    }
    if (radius == 0.0f) radius = 1.0f;
    float inv_rad = 1.0f / radius;
    for (auto& v : gVertices) {
        v.pos = (v.pos - center) * inv_rad;
    }

    std::cout << "Loaded and normalized " << gVertices.size() << " total vertices" << std::endl;
}
