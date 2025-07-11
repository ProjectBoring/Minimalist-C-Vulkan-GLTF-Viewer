You can use these to manipulate shaders, lighting, colors, etc.

Editable:
- Edit viewer.frag
- Edit viewer.vert

You can edit those two files and then build them like so with cmd prompt:


```
cd .\shaders

"C:\VulkanSDK\1.4.313.2\Bin\glslc.exe" viewer.vert -o viewer.vert.spv

"C:\VulkanSDK\1.4.313.2\Bin\glslc.exe" viewer.frag -o viewer.frag.spv
```