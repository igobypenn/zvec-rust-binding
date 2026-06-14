# Overlay CMakeLists that compiles upstream zvec c_api.cc as a STATIC library.
# This avoids the SHARED-only default in upstream src/binding/c/CMakeLists.txt
# and preserves the project's static-linking deployment story.
cmake_minimum_required(VERSION 3.13)
project(zvec_c_api_static CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ZVEC_SRC)
    message(FATAL_ERROR "ZVEC_SRC must point at the upstream zvec source directory")
endif()
if(NOT DEFINED ZVEC_BUILD)
    message(FATAL_ERROR "ZVEC_BUILD must point at the upstream zvec build directory")
endif()

add_library(zvec_c_api_static STATIC
    ${ZVEC_SRC}/src/binding/c/c_api.cc
)

target_include_directories(zvec_c_api_static PRIVATE
    ${ZVEC_BUILD}/src/generated                 # configured zvec/c_api.h and zvec_version.h
    ${ZVEC_SRC}/src/include                     # zvec/db/*.h, zvec/c_api.h source
    ${ZVEC_SRC}/src                             # internal headers
)

set_target_properties(zvec_c_api_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden                # mirror upstream: hide non-ZVEC_EXPORT symbols
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "zvec_c_api_static"
)

target_compile_options(zvec_c_api_static PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wno-unused-parameter>
)
