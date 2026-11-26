cmake_minimum_required(VERSION 3.28)
project(Lab_03 LANGUAGES CXX CUDA)

# Стандарты C++ и CUDA
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ==============================================================================
# Настройка архитектуры GPU
# ==============================================================================
# Compute Capability:
# Turing = 7.5, Ampere = 8.6, Ada = 8.9, Blackwell = 12.0
# ==============================================================================

# ==============================================================================
# Настройка архитектуры GPU
# ==============================================================================
set(LAB_CUDA_ARCHS "" CACHE STRING
    "CUDA architectures, e.g. 89 or '80;89'. Empty = auto-detect")

option(MATRIX_NATIVE_CPU
    "Use -march=native for host code (disable if building on login node)" ON)

set(CUDA_ARCH_CANDIDATES 50 52 53 60 61 62 70 72 75 80 86 87 89 90 120)

execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
    OUTPUT_VARIABLE CUDA_SUPPORTED_ARCHS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE CUDA_ARCH_ERROR
    RESULT_VARIABLE CUDA_ARCH_RESULT
)

if(NOT CUDA_ARCH_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Unable to determine supported CUDA architectures:\n${CUDA_ARCH_ERROR}"
    )
endif()

if(NOT CUDA_SUPPORTED_ARCHS MATCHES "compute_[0-9]+")
    message(FATAL_ERROR
        "NVCC returned an unexpected architecture list:\n${CUDA_SUPPORTED_ARCHS}"
    )
endif()

if(LAB_CUDA_ARCHS)
    set(CMAKE_CUDA_ARCHITECTURES ${LAB_CUDA_ARCHS})

    foreach(ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(NOT CUDA_SUPPORTED_ARCHS MATCHES "compute_${ARCH}([^0-9]|$)")
            message(FATAL_ERROR
                "CUDA architecture ${ARCH} is not supported by current NVCC."
            )
        endif()
    endforeach()
else()
    set(CUDA_ARCH_LIST "")

    foreach(ARCH IN LISTS CUDA_ARCH_CANDIDATES)
        if(CUDA_SUPPORTED_ARCHS MATCHES "compute_${ARCH}([^0-9]|$)")
            list(APPEND CUDA_ARCH_LIST ${ARCH})
        else()
            message(STATUS
                "CUDA architecture ${ARCH} is not supported by current NVCC. Skipping."
            )
        endif()
    endforeach()

    if(NOT CUDA_ARCH_LIST)
        message(FATAL_ERROR "No supported CUDA architectures found!")
    endif()

    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
endif()

message(STATUS "Selected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ==============================================================================
# Поиск зависимостей CUDA Toolkit (включает cudart и cublas)
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# Precision configuration
option(MATRIX_USE_DOUBLE "Use double precision instead of float" OFF)

# Optional fast math; disabled by default to preserve numerical accuracy
option(MATRIX_USE_FAST_MATH "Enable CUDA fast math" OFF)

# Documentation
option(BUILD_DOC "Build Doxygen documentation" ON)

# ==============================================================================
# Цели сборки (Targets)
# ==============================================================================
set(SOURCE_FILES matrix_kernels.cu benchmark.cu main.cu)
add_executable(Lab_03 ${SOURCE_FILES})

# Подключаем библиотеки CUDA Runtime и cuBLAS
target_link_libraries(Lab_03 PRIVATE
        CUDA::cudart
        CUDA::cublas
)

target_include_directories(Lab_03 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Apply precision setting consistently to all translation units
if(MATRIX_USE_DOUBLE)
    target_compile_definitions(Lab_03 PRIVATE MATRIX_USE_DOUBLE=1)
else()
    target_compile_definitions(Lab_03 PRIVATE MATRIX_USE_DOUBLE=0)
endif()

set(MATRIX_TILE_SIZE 32 CACHE STRING "CUDA tile/block dimension")
set_property(CACHE MATRIX_TILE_SIZE PROPERTY STRINGS 2 4 8 16 32)

if (NOT MATRIX_TILE_SIZE MATCHES "^(2|4|8|16|32)$")
    message(FATAL_ERROR
        "Unsupported MATRIX_TILE_SIZE=${MATRIX_TILE_SIZE}. "
        "Allowed values: 2, 4, 8, 16, 32."
    )
endif()

target_compile_definitions(
    Lab_03 PRIVATE
    MATRIX_TILE_SIZE=${MATRIX_TILE_SIZE}
)

# ==============================================================================
# Оптимизация для Release (используем target_compile_options)
# ==============================================================================
if(MSVC)
    target_compile_definitions(Lab_03 PRIVATE NOMINMAX)

    target_compile_options(Lab_03 PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:/Ob2>
        $<$<COMPILE_LANGUAGE:CXX>:/W4>

        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
else()
    target_compile_options(Lab_03 PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>
         $<$<AND:$<BOOL:${MATRIX_NATIVE_CPU}>,$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>

        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    )
endif()

if(MATRIX_USE_FAST_MATH)
    target_compile_options(Lab_03 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif()

message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
if(MATRIX_USE_DOUBLE)
    message(STATUS "Matrix precision: double")
else()
    message(STATUS "Matrix precision: float")
endif()
message(STATUS "Fast math: ${MATRIX_USE_FAST_MATH}")

if(BUILD_DOC)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)
        if(EXISTS ${DOXYGEN_IN})
            configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
            add_custom_target(doc ALL
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    COMMENT "Generating Doxygen documentation"
                    VERBATIM
            )
            message(STATUS "✓ Doxygen found: ${DOXYGEN_EXECUTABLE}")
            message(STATUS "  Build docs with: cmake --build build --target doc")
        else()
            message(WARNING "⚠ Doxyfile not found in source directory. Documentation target skipped.")
        endif()
    else()
        message(WARNING "⚠ Doxygen not found. Install it or set DOXYGEN_EXECUTABLE path.")
        message(WARNING "  Windows: winget install Doxygen.Doxygen")
        message(WARNING "  Or specify path: cmake -B build -DDOXYGEN_EXECUTABLE=\"C:/Program Files/doxygen/bin/doxygen.exe\"")
    endif()
endif()
