cmake_minimum_required(VERSION 3.28)
project(Lab_02 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Оптимизация для Release
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # GCC / Clang (Linux, macOS, MinGW)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-O3 -march=native -pedantic -Wall -Wextra -Werror)
    elseif(MSVC)
        add_compile_options(/O2 /Ob2 /W4)
        add_compile_definitions(NOMINMAX)
    endif()
endif()

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# 1. Основная версия (с внутренним профилированием через MPI_Wtime)
set(SOURCE_FILES main.cpp poisson_solver.hpp)
add_executable(Lab_02 ${SOURCE_FILES})
target_link_libraries(Lab_02 PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# 2. Версия для MPE профилирования (автоматически добавляется, если файл существует)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/main_mpe_jumpshot.cpp")
    set(SOURCE_MPE_FILES main_mpe_jumpshot.cpp poisson_solver.hpp)
    add_executable(Lab_02_mpe ${SOURCE_MPE_FILES})
    target_compile_definitions(Lab_02_mpe PRIVATE USE_MPE)
    target_link_libraries(Lab_02_mpe PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX mpe)
    target_include_directories(Lab_02_mpe PRIVATE /usr/local/include)
    target_link_directories(Lab_02_mpe PRIVATE /usr/local/lib)
    message(STATUS "✓ Found main_mpe_jumpshot.cpp: Target 'Lab_02_mpe' will be built with MPE support.")
endif()

if(MSVC)
    target_compile_options(Lab_02 PRIVATE /openmp:llvm)
endif()

message(STATUS "MPI Found: ${MPI_CXX_COMPILER}")
message(STATUS "MPI Include: ${MPI_CXX_INCLUDE_DIRS}")
message(STATUS "MPI Libraries: ${MPI_CXX_LIBRARIES}")
message(STATUS "OpenMP Found: ${OpenMP_CXX_FLAGS}")

option(BUILD_DOC "Build Doxygen documentation" ON)

if(BUILD_DOC)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)
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
        message(WARNING "⚠ Doxygen not found. Install it or set DOXYGEN_EXECUTABLE path.")
        message(WARNING "  Windows: winget install Doxygen.Doxygen")
        message(WARNING "  Or specify path: cmake -B build -DDOXYGEN_EXECUTABLE=\"C:/Program Files/doxygen/bin/doxygen.exe\"")
    endif()
endif()
