file(GLOB VLFEAT_SRCS vl/*.c vl/*.h)

# Add compiler-specific warning suppression flags
if(MSVC)
    # implicit conversion loss of data (closest match to -Wno-absolute-value)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4244") 
  # initialized but unreferenced variable (-Wno-unused-but-set-variable)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4189")  
  # operator precedence warnings (-Wno-logical-not-parentheses)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4554") 
  # expression is always false/true (-Wno-tautological-constant-out-of-range-compare)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4296")
else()
  # GCC/Clang: disable specific warnings
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-absolute-value")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-but-set-variable")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-logical-not-parentheses")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-tautological-constant-out-of-range-compare")
endif()

# vlfeat SSE2 disabled on arm
if( (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") OR (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64") )
    add_definitions( -DVL_DISABLE_SSE2 )
endif()

if(WIN32)
    add_definitions(-D__SSE2__)
endif()

add_library(vl ${VLFEAT_SRCS})


target_include_directories(vl
  PRIVATE
    ${CMAKE_SOURCE_DIR}/third_party/vlfeat
    ${OpenMP_CXX_INCLUDE_DIR})
