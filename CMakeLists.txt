cmake_minimum_required(VERSION 2.8)
project(kpts_line_search)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/" ${CMAKE_MODULE_PATH})

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wextra -pedantic -fopenmp")

add_definitions(-std=c++11)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
set(${PYTHONHOME} ${PYTHON_INCLUDE_DIRS})
include_directories(${PYTHONHOME}/include)
include_directories(${PYTHONHOME}/Lib/site-packages/numpy/core/include)
link_directories(${PYTHONHOME}/libs)
add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)
include_directories(include/plot_tools)

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3.4.6 REQUIRED)

find_package(TBB REQUIRED)

message(${TBB_INCLUDE_DIR})
message(${TBB_LIBRARIES})


FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(Pangolin REQUIRED)
find_package(PCL 1.8 REQUIRED)
include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}

        ${OpenMP_INCLUDE_DIRS} #多线程库
        ${TBB_INCLUDE_DIR}
        include
        src
)

find_package(Boost COMPONENTS filesystem program_options system REQUIRED)
link_directories(${Boost_LIBRARY_DIRS})

FILE(GLOB SRC_LIST_CPP
        src/*.cpp
        )


#TODO 关键点线搜索--------------------------------------------------------------------------
add_executable(kpts_line_search
        main_kpts_est.cpp
        src/tool.cpp
        src/keypoints.cpp
        ${SRC_LIST_CPP}
        )

target_link_libraries(kpts_line_search
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        )











