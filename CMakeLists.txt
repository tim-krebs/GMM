cmake_minimum_required(VERSION 3.12)

set(PROJECT "SpeechRecognition")

project("${PROJECT}_Project" VERSION 1.0.0 LANGUAGES CXX)

add_library(DataHandler     OBJECT     ${PROJECT_SOURCE_DIR}/src/DataHandler.cpp)


include_directories(Timer           ${PROJECT_SOURCE_DIR}/include/)
include_directories(Filter          ${PROJECT_SOURCE_DIR}/include/)
include_directories(Matrix          ${PROJECT_SOURCE_DIR}/include/)

# Global CMake variables are set here
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)
set (CONAN_DISABLE_CHECK_COMPILER TRUE)

# Cmake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)
include(Documentation)
if(ENABLE_TESTING)
    include(CTest)
endif()

# Project/Library Names
set(LIBRARY_NAME    "${PROJECT}_MyLib")
set(MAIN            "${PROJECT}_Executable")

#Add conan libraries
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Options for generating/building
# cd build && cmake -D USE_PRINTF=[OFF|ON] ..
option(USE_PRINTF "If you want either to use printf (ON) or cout (OFF)" OFF)

# Configuration Files
configure_file("${PROJECT_SOURCE_DIR}/include/ProjectConfig.h.in"
               "${PROJECT_BINARY_DIR}/ProjectConfig.h" @ONLY)

# Add libraries
add_subdirectory("${PROJECT_SOURCE_DIR}/src/")

# Add executables
add_subdirectory("${PROJECT_SOURCE_DIR}/app/")
