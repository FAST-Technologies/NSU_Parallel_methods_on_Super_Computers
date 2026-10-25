 find_package(OpenMP REQUIRED)

 add_executable(minirt-test minirt_test.cpp)

target_link_libraries(minirt-test PUBLIC minirt OpenMP::OpenMP_CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#install(TARGETS minirt-test DESTINATION bin/minirt)
