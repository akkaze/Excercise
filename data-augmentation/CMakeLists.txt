cmake_minimum_required(VERSION 2.8)
set(project generator)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/module)
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=gnu++0x)
endif()

SET(CMAKE_BUILD_TYPE Debug)
SET(SRC_LIST "main.cpp")
FIND_PACKAGE(OpenCV REQUIRED)
find_package(Boost COMPONENTS chrono thread date_time program_options filesystem system REQUIRED)
SET(BOOST_DIR "/usr/include")

find_package(GFlags REQUIRED)
ADD_EXECUTABLE(${project} ${SRC_LIST})
include_directories(${BOOST_LIBRARYDIR} ${OpenCV_INCLUDE_PATH} ${GFLAGS_INCLUDE_DIRS})

target_link_libraries(${project} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${GFLAGS_LIBRARIES})
