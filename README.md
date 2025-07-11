# Minimalist C++ Vulkan glTF Viewer

---

This project is a minimalist C++ Vulkan viewer designed to load and render glTF 2.0 models using GLFW for window management and `fastgltf` for efficient glTF parsing. It's built with a focus on core Vulkan concepts, including swapchain management, depth buffering, and a basic rendering pipeline with push constants for model transformations.

## Features

* **Vulkan Core Integration:** Initializes Vulkan instance, physical device, logical device, swapchain, render pass, and graphics pipeline.
* **Window Management:** Uses GLFW for creating a Vulkan-compatible window and handling events like resizing, mouse input (for model manipulation), and scrolling.
* **glTF 2.0 Loading:** Employs `fastgltf` to parse glTF files, including:
    * Loading vertex data (positions, normals, UVs).
    * Handling indexed drawing.
    * Loading and applying PBR materials with base color textures.
    * Supports loading glTF from local paths, URLs, and Windows resources (DLLs/EXEs).
* **GPU-Driven Transformations:** Model rotation, pitch, and zoom are applied directly on the GPU using push constants, optimizing performance.
* **Device-Local Buffers:** Utilizes device-local vertex and index buffers for optimal rendering performance.
* **Texture Support:** Loads image data (e.g., PNG, JPG) using `stb_image` and creates Vulkan image views and samplers for textured materials.
* **Basic Scene Graph:** Processes glTF nodes and meshes, though currently, it combines all primitives into a single mesh for simplicity.
* **Model Normalization:** Automatically centers and scales loaded glTF models to fit within a unit sphere, ensuring consistent viewing regardless of model size.

## Screenshots

*Capabilities:*

<img width="402" height="323" alt="image" src="https://github.com/user-attachments/assets/9367f236-40bb-4110-a41f-f5ffa35a1698" />
<img width="402" height="323" alt="image" src="https://github.com/user-attachments/assets/eae6afa6-012a-4797-ba40-73191743e830" />

## Getting Started

### Prerequisites

* A C++ compiler (e.g., MSVC on Windows, GCC/Clang on Linux/macOS)
* [Vulkan SDK](https://vulkan.lunarg.com/sdk/home)
* [GLFW](https://www.glfw.org/download.html) (Development files)
* [glm](https://github.com/g-truc/glm)
* [stb_image](https://github.com/nothings/stb)
* [fastgltf](https://github.com/KhronosGroup/glTF-Parser)
* [libcurl](https://curl.se/libcurl/) (for URL loading)

### Building the Project

This project uses embedded SPIR-V shaders (`viewer_vert_spv.h` and `viewer_frag_spv.h`). These files are typically generated from `.vert` and `.frag` source files using the Vulkan SDK's `glslangValidator`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ProjectBoring/Minimalist-C-Vulkan-GLTF-Viewer.git
    cd Minimalist-C-Vulkan-GLTF-Viewer
    ```
2.  **Acquire Dependencies:**
    * Download and install the **Vulkan SDK**.
    * Download the **GLFW** source or pre-compiled binaries. You'll need to link against `glfw3.lib` (or equivalent).
    * For **glm**, **stb_image**, and **fastgltf**, you can often include their header files directly into your project. For `fastgltf` specifically, ensure you also link its compiled library if you're not including it as a header-only library.
    * For **libcurl**, download its development package and link the appropriate library (`libcurl.lib` or equivalent).
    * Ensure your build system or IDE's project settings include the header paths for these libraries and link against their respective `.lib` or `.a` files.
3.  **Generate Shader Headers:** If your shaders change, or if they are not already generated, run the appropriate `glslangValidator` commands to generate `viewer_vert_spv.h` and `viewer_frag_spv.h` from your shader source files (e.g., `viewer.vert` and `viewer.frag`). For example, from your shader directory:
    ```bash
    # Assuming glslangValidator is in your PATH or you provide its full path
    glslangValidator -V viewer.vert -o viewer_vert_spv.h -H
    glslangValidator -V viewer.frag -o viewer_frag_spv.h -H
    ```
    *(Adjust `viewer.vert` and `viewer.frag` to your actual shader file names if different, and ensure the output headers are placed where your C++ files can find them.)*
4.  **Compile:** Compile the project using your chosen C++ compiler or IDE. You will need to configure your project to:
    * Include paths for Vulkan, GLFW, glm, stb_image, fastgltf, and libcurl headers.
    * Link against the necessary Vulkan, GLFW, fastgltf, and libcurl libraries.

### Running the Viewer

After building, you can run the executable. By default, it attempts to load a glTF model from `GLTF_PATH` or `GLTF_URL` specified in `Main.cpp`, or from a Windows resource `IDR_GLTF1` if on Windows.

**Important:** Remember to change `GLTF_PATH` or `GLTF_URL` in `Main.cpp` to your desired glTF model's path or URL. If loading from a Windows resource, ensure `resource.h` and the `.rc` file correctly embed your glTF.

```cpp
const char* GLTF_PATH = "C:/Users/admin/Downloads/MyAvatar.gltf"; // ❗ CHANGE TO YOUR MODEL
const char* GLTF_URL = "https://raw.githubusercontent.com/SaschaWillems/Vulkan-glTF-PBR/refs/heads/master/data/models/DamagedHelmet/glTF-Embedded/DamagedHelmet.gltf"; // ❗ CHANGE TO YOUR MODEL
