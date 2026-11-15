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
# Blackwell (RTX 50xx) = 120, Ada (RTX 40xx) = 89, Ampere = 86, Turing = 75
set(CMAKE_CUDA_ARCHITECTURES 75 86 89 120)

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
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
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
