#
#
#
#  Created by David Chiu
#  Dec. 28th, 2018 based on tencent's original file
#
#
project(Imageutil)
cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE on)

#get_filename_component(TF_SRC_ROOT ${CMAKE_SOURCE_DIR}/../../../..  ABSOLUTE)
get_filename_component(IMGUTIL_SRC_DIR  ${CMAKE_SOURCE_DIR}  ABSOLUTE)
get_filename_component (NCNN_JNI_SOURCE_DIR ${CMAKE_SOURCE_DIR}/ncnn  ABSOLUTE)

if (ANDROID_ABI MATCHES "^armeabi-v7a$")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=softfp -mfpu=neon -fopenmp")
elseif(ANDROID_ABI MATCHES "^arm64-v8a")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ftree-vectorize -fopenmp")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTANDALONE_DEMO_LIB \
                    -std=c++11 -fno-exceptions -fno-rtti -O2 -Wno-narrowing \
                    -fPIE")

add_definitions(-DHEADER_FILE=${HEADER_FILE})
add_definitions(-DNAMESPACE=${NAMESPACE})

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
                              -Wl,--allow-multiple-definition \
                              -Wl,--whole-archive -fPIE -v -fopenmp")

file(GLOB_RECURSE imgutil_sources ${IMGUTIL_SRC_DIR}/*.cc)

add_library(imgutil SHARED
            ${imgutil_sources})
target_include_directories(imgutil PRIVATE
                           ${CMAKE_SOURCE_DIR}
                           )

target_link_libraries(imgutil
                      android
                      log
                      jnigraphics
                      m
                      atomic
                      z)
