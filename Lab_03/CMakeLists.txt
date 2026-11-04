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
# Поиск зависимостей CUDA Toolkit (включает cudart и cublas)
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# Цели сборки (Targets)
# ==============================================================================
set(SOURCE_FILES matrix_kernels.cu main.cu)
add_executable(Lab_03 ${SOURCE_FILES})

# Подключаем библиотеки CUDA Runtime и cuBLAS
target_link_libraries(Lab_03 PRIVATE
        CUDA::cudart
        CUDA::cublas
)

if(MSVC)
    target_compile_options(Lab_03 PRIVATE /openmp:llvm)
endif()

target_include_directories(Lab_03 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# ==============================================================================
# Оптимизация для Release (используем target_compile_options)
# ==============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Флаги для C++ (GCC / Clang)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(Lab_03 PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Werror>
        )
    elseif(MSVC)
        target_compile_options(Lab_03 PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2 /Ob2 /W4>
        )
        target_compile_definitions(Lab_03 PRIVATE NOMINMAX)
    endif()

    # Флаги для CUDA (nvcc)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(Lab_03 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
        )
    elseif(MSVC)
        target_compile_options(Lab_03 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
        )
    endif()
endif()

if(MSVC)
    target_compile_options(Lab_03 PRIVATE /openmp:llvm)
endif()

message(STATUS "✓ CUDA Toolkit Found: ${CUDAToolkit_VERSION}")
message(STATUS "✓ CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "✓ Build Type: ${CMAKE_BUILD_TYPE}")

option(BUILD_DOC "Build Doxygen documentation" ON)

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
