cmake_minimum_required(VERSION 3.20)

project(Lab_04 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)

add_executable(dwsp dwsp.cpp dwsp.hpp)

target_link_libraries(dwsp PRIVATE MPI::MPI_CXX Threads::Threads)
target_include_directories(dwsp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(dwsp PRIVATE -Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(dwsp PRIVATE -O2)
    endif()
elseif(MSVC)
    target_compile_options(dwsp PRIVATE /W4)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(dwsp PRIVATE /O2)
    endif()
endif()

if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI")
    target_compile_definitions(dwsp PRIVATE OMPI_SKIP_MPICXX)
endif()

option(BUILD_DOC "Build HTML documentation with Doxygen" ON)

if(BUILD_DOC)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
        set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")

        if(EXISTS "${DOXYGEN_IN}")
            configure_file("${DOXYGEN_IN}" "${DOXYGEN_OUT}" @ONLY)
            add_custom_target(doc ALL
                    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_OUT}"
                    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
                    COMMENT "Generating Doxygen documentation"
                    VERBATIM
            )
            message(STATUS "✓ Doxygen found: ${DOXYGEN_EXECUTABLE}")
            message(STATUS "  Build docs with: cmake --build build --target doc")
        else()
            message(WARNING "⚠ Doxyfile not found in source directory. Documentation target skipped.")
        endif()
    else()
        message(WARNING "⚠ Doxygen not found. Install it  with: sudo apt install doxygen graphviz or set DOXYGEN_EXECUTABLE path.")
        message(WARNING "  Windows: winget install Doxygen.Doxygen")
        message(WARNING "  Or specify path: cmake -B build -DDOXYGEN_EXECUTABLE=\"C:/Program Files/doxygen/bin/doxygen.exe\"")
    endif()
endif()

message(STATUS "DWSP C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "DWSP build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "MPI version: ${MPI_CXX_VERSION}")
message(STATUS "Doxygen enabled: ${BUILD_DOC}")

