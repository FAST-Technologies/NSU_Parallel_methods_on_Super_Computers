cmake_minimum_required(VERSION 3.29)
project(Lab_01 LANGUAGES CXX)

# Стандарт C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Компилятор и флаги
# ============================================================
if(MSVC)
    # Включаем современный OpenMP через LLVM-бэкенд MSVC
    # Доступно в Visual Studio 2022 17.10+
    add_compile_options(/utf-8 /Zc:__cplusplus /W4)
    add_compile_definitions(NOMINMAX)

    # Оптимизация для Release
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2")
else()
    # GCC / Clang (Linux, macOS, MinGW)
    add_compile_options(-O3 -Wall -Wextra -fopenmp)
    add_link_options(-fopenmp)
endif()

# ============================================================
# OpenMP
# ============================================================
find_package(OpenMP REQUIRED)

# Целевой исполняемый файл
set(SOURCE_FILES main.cpp)
add_executable(Lab_01 ${SOURCE_FILES})

message(STATUS "Checking OpenMP")
# Для MSVC добавляем современный OpenMP через LLVM-бэкенд
if(MSVC)
    target_compile_options(Lab_01 PRIVATE /openmp:llvm)
endif()
# Линковка OpenMP (автоматически добавляет /openmp для MSVC)
target_link_libraries(Lab_01 PRIVATE OpenMP::OpenMP_CXX)

# ============================================================================
# Опциональная генерация Doxygen-документации
# ============================================================================
option(BUILD_DOC "Build Doxygen documentation" ON)

if(BUILD_DOC)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

        # Конфигурируем Doxyfile (подставляем пути)
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

        # Цель для генерации документации
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

