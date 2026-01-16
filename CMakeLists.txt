cmake_minimum_required(VERSION 3.10)
project(stable_diffusion_cpp)

set(CMAKE_CXX_STANDARD 17)

# Set OpenCV directory
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv/build)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(sd
    main.cpp
    onnx_model.cpp
    lcm_scheduler.cpp
    clip_tokenizer.cpp
    clip_tokenizer_complete.cpp
    stable_diffusion_pipeline.cpp
)

# Link libraries
target_link_libraries(sd
    ${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2/lib/onnxruntime.lib
    ${OpenCV_LIBS}
)

# Copy DLLs to output directory (Windows)
if(WIN32)
    add_custom_command(TARGET sd POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll"
        "${CMAKE_SOURCE_DIR}/onnxruntime-win-x64-1.23.2/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:sd>
    )
endif()