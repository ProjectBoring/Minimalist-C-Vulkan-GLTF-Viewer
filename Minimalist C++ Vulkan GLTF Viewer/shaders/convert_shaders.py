import os

def bin_to_c_array(input_path, output_path, array_name):
    with open(input_path, 'rb') as f_in:
        byte_data = f_in.read()

    with open(output_path, 'w') as f_out:
        f_out.write(f'#pragma once\n\n')
        f_out.write(f'// Generated from {os.path.basename(input_path)}\n')
        f_out.write(f'unsigned char {array_name}[] = {{\n')
        
        for i, byte in enumerate(byte_data):
            f_out.write(f'0x{byte:02x}')
            if i < len(byte_data) - 1:
                f_out.write(', ')
            if (i + 1) % 16 == 0:  # New line after every 16 bytes
                f_out.write('\n')
        
        f_out.write('\n};\n')
        f_out.write(f'unsigned int {array_name}_size = {len(byte_data)};\n')

if __name__ == "__main__":
    # Assuming the script is run from the directory containing viewer.vert.spv and viewer.frag.spv
    # For your project, this would be 'C:/Users/admin/source/repos/Minimalist C++ Vulkan GLTF Viewer/x64/Debug/'
    current_dir = os.path.dirname(os.path.abspath(__file__)) 

    vert_spv_path = os.path.join(current_dir, "viewer.vert.spv")
    frag_spv_path = os.path.join(current_dir, "viewer.frag.spv")

    bin_to_c_array(vert_spv_path, os.path.join(current_dir, "viewer_vert_spv.h"), "g_viewer_vert_spv")
    bin_to_c_array(frag_spv_path, os.path.join(current_dir, "viewer_frag_spv.h"), "g_viewer_frag_spv")

    print("Shader binaries converted to C++ header files.")