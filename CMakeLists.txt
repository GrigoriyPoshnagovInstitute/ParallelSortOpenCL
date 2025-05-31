cmake_minimum_required(VERSION 3.12)
project(ParallelSortOpenCL)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "C:/OpenCLDev/OpenCL-SDK/install")

find_package(OpenCLHeaders REQUIRED)
find_package(OpenCLHeadersCpp REQUIRED)
find_package(OpenCL REQUIRED)

add_executable(ParallelSortOpenCL
        main.cpp
)

# 🔽 Добавлено: копируем BitonicSort.cl в папку сборки
configure_file(BitonicSort.cl BitonicSort.cl COPYONLY)

target_compile_definitions(ParallelSortOpenCL PRIVATE
        CL_TARGET_OPENCL_VERSION=200
        CL_HPP_TARGET_OPENCL_VERSION=200
)

target_link_libraries(ParallelSortOpenCL PRIVATE
        OpenCL::OpenCL
        OpenCL::Headers
        OpenCL::HeadersCpp
)

if (MSVC)
    target_compile_options(ParallelSortOpenCL PRIVATE /W4 /WX)
else()
    target_compile_options(ParallelSortOpenCL PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(WIN32)
    set(OPENCL_SDK_DLL_PATH "C:/OpenCLDev/OpenCL-SDK/install/bin/OpenCL.dll")
    if(EXISTS "${OPENCL_SDK_DLL_PATH}")
        add_custom_command(TARGET ParallelSortOpenCL POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OPENCL_SDK_DLL_PATH}"
                $<TARGET_FILE_DIR:ParallelSortOpenCL>
                COMMENT "Copying OpenCL.dll from SDK to build directory"
        )
    else()
        message(WARNING "OpenCL.dll from SDK not found at ${OPENCL_SDK_DLL_PATH}.")
    endif()
endif()
