cmake_minimum_required(VERSION 3.24)
project(VLLM_SERVER LANGUAGES CXX)

# Suppress FindBoost warning (Boost is now config-mode only in CMake 3.30+)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES 
      src/base_service.cpp
      src/device_worker.cpp
      src/runner.cpp
      src/scheduler.cpp
      src/server.cpp
      src/tracker.cpp
)

include(FetchContent)

# Required packages - always required
find_package(glog REQUIRED)
find_package(Threads REQUIRED)

# fmt - try system first, fetch if not found
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  message(STATUS "fmt not found, fetching from GitHub...")
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 10.2.1
  )
  FetchContent_MakeAvailable(fmt)
endif()

# Boost - only needed for interprocess message queues
find_package(Boost 1.64 QUIET COMPONENTS system)
if(NOT Boost_FOUND)
  message(STATUS "Boost not found, fetching from GitHub...")
  set(BOOST_INCLUDE_LIBRARIES interprocess system)
  set(BOOST_ENABLE_CMAKE ON)
  FetchContent_Declare(
    Boost
    GIT_REPOSITORY https://github.com/boostorg/boost.git
    GIT_TAG boost-1.84.0
    GIT_SHALLOW TRUE
    OVERRIDE_FIND_PACKAGE
  )
  FetchContent_MakeAvailable(Boost)
  set(BOOST_LIBS Boost::interprocess)
else()
  message(STATUS "Found system Boost: ${Boost_VERSION}")
  set(BOOST_LIBS Boost::headers)
  if(Boost_VERSION_STRING VERSION_LESS "1.69")
    list(APPEND BOOST_LIBS Boost::system)
  endif()
endif()

# Drogon - HTTP framework with proper chunked streaming support
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_CTL OFF CACHE BOOL "" FORCE)
set(BUILD_ORM OFF CACHE BOOL "" FORCE)
set(BUILD_BROTLI OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  drogon
  GIT_REPOSITORY https://github.com/drogonframework/drogon.git
  GIT_TAG v1.9.8
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(drogon)

add_executable(vllm_server ${SOURCES})

target_include_directories(vllm_server 
  PRIVATE 
    inc
)

target_link_libraries(vllm_server 
  PRIVATE
    glog::glog 
    fmt::fmt
    ${BOOST_LIBS}
    Threads::Threads
    drogon
)
