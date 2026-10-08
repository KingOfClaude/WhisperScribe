cmake_minimum_required(VERSION 3.18)
project(WhisperScribe CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# ---------------------------------------------------------------- CUDA (Windows)
# Only active when configured with -DGGML_CUDA=ON (build-cuda.ps1 does this).
# CUDA 13's CCCL headers need MSVC's standard-conforming preprocessor. It has to reach
# cl.exe through -Xcompiler (a bare flag would be misread by nvcc as an input file).
#
# NOTE: do NOT hard-code CMAKE_CUDA_ARCHITECTURES here. Kernels built for the wrong GPU
# architecture fail at runtime with "no kernel image is available for execution on the
# device". build-cuda.ps1 detects your GPU and passes the right value; if you configure
# by hand, pass -DCMAKE_CUDA_ARCHITECTURES=native (or e.g. 86) yourself.
if(MSVC AND GGML_CUDA)
  if(NOT CMAKE_CUDA_FLAGS MATCHES "Zc:preprocessor")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor"
        CACHE STRING "CUDA compiler flags" FORCE)
  endif()
endif()

# ---------------------------------------------------------------- whisper.cpp
# GPU: configure with -DGGML_CUDA=ON (NVIDIA), -DGGML_VULKAN=ON, or on macOS
# Metal is enabled automatically.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(whisper
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG        v1.7.5)

# ---------------------------------------------------------------------- GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)  # runs fine via XWayland
FetchContent_Declare(glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4)

# ------------------------------------------------- native file dialogs (NFD)
set(NFD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nfd
  GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
  GIT_TAG        v1.2.1)

# ------------------------------------------- Dear ImGui & miniaudio (sources)
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.9
  SOURCE_SUBDIR  _no_cmake_here)
FetchContent_Declare(miniaudio
  GIT_REPOSITORY https://github.com/mackron/miniaudio.git
  GIT_TAG        0.11.21
  SOURCE_SUBDIR  _no_cmake_here)

FetchContent_MakeAvailable(whisper glfw nfd imgui miniaudio)

# ------------------------------------------- Parakeet engine via sherpa-onnx (optional)
# Uses the prebuilt *shared* sherpa-onnx package (no compiling). Turn off with
# -DWITH_PARAKEET=OFF. To use a package you extracted yourself: -DSHERPA_ONNX_ROOT=<folder>
option(WITH_PARAKEET "Add the NVIDIA Parakeet engine (prebuilt sherpa-onnx)" ON)
set(SHERPA_ONNX_VERSION "1.13.1" CACHE STRING "sherpa-onnx release to download")
set(SHERPA_ONNX_ROOT "" CACHE PATH "Already-extracted sherpa-onnx shared package (skips the download)")

if(WITH_PARAKEET)
  set(_sherpa_root "${SHERPA_ONNX_ROOT}")
  if(NOT _sherpa_root)
    set(_v "${SHERPA_ONNX_VERSION}")
    if(WIN32)
      set(_sherpa_archive "sherpa-onnx-v${_v}-win-x64-shared-MT-Release-lib.tar.bz2")
    elseif(APPLE)
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_sherpa_archive "sherpa-onnx-v${_v}-osx-arm64-shared-lib.tar.bz2")
      else()
        set(_sherpa_archive "sherpa-onnx-v${_v}-osx-x64-shared-lib.tar.bz2")
      endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(_sherpa_archive "sherpa-onnx-v${_v}-linux-x64-shared-lib.tar.bz2")
    else()
      message(FATAL_ERROR "No prebuilt sherpa-onnx for this platform. Set -DSHERPA_ONNX_ROOT=... or -DWITH_PARAKEET=OFF")
    endif()
    FetchContent_Declare(sherpa_onnx
      URL "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${_v}/${_sherpa_archive}"
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
      SOURCE_SUBDIR _no_cmake_here)
    FetchContent_MakeAvailable(sherpa_onnx)
    set(_sherpa_root "${sherpa_onnx_SOURCE_DIR}")
  endif()

  # import library (layout-agnostic search)
  file(GLOB_RECURSE _sherpa_libs
    "${_sherpa_root}/*sherpa-onnx-c-api.lib"
    "${_sherpa_root}/*libsherpa-onnx-c-api.so"
    "${_sherpa_root}/*libsherpa-onnx-c-api.dylib")
  if(NOT _sherpa_libs)
    message(FATAL_ERROR "sherpa-onnx-c-api library not found under ${_sherpa_root}.\n"
                        "Check the package layout, or build with -DWITH_PARAKEET=OFF.")
  endif()
  list(GET _sherpa_libs 0 SHERPA_C_API_LIB)

  # header: use the one shipped in the package, else fetch the matching one for this tag
  file(GLOB_RECURSE _sherpa_hdrs "${_sherpa_root}/*c-api.h")
  list(FILTER _sherpa_hdrs INCLUDE REGEX "sherpa-onnx/c-api/c-api\\.h$")
  if(_sherpa_hdrs)
    list(GET _sherpa_hdrs 0 _h)
    get_filename_component(_d "${_h}" DIRECTORY)
    get_filename_component(_d "${_d}" DIRECTORY)
    get_filename_component(SHERPA_INCLUDE_DIR "${_d}" DIRECTORY)
  else()
    set(SHERPA_INCLUDE_DIR "${CMAKE_BINARY_DIR}/sherpa_headers")
    file(DOWNLOAD
      "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v${SHERPA_ONNX_VERSION}/sherpa-onnx/c-api/c-api.h"
      "${SHERPA_INCLUDE_DIR}/sherpa-onnx/c-api/c-api.h"
      STATUS _dl_status)
    list(GET _dl_status 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
      message(FATAL_ERROR "Could not download sherpa-onnx c-api.h (${_dl_status}).")
    endif()
  endif()
  message(STATUS "sherpa-onnx: lib=${SHERPA_C_API_LIB}  include=${SHERPA_INCLUDE_DIR}")
endif()

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/misc/cpp/imgui_stdlib.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_lib PUBLIC
  ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${imgui_SOURCE_DIR}/misc/cpp)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL)

add_executable(whisper_scribe main.cpp downloader.cpp)

# Windows: embed the app icon + version info (app.rc) so the exe, taskbar and Alt+Tab show it
if(WIN32)
  enable_language(RC)
  target_sources(whisper_scribe PRIVATE app.rc)
  set_source_files_properties(app.rc PROPERTIES
    OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.ico")
endif()
if(MSVC)
  # only for our own C++ sources: a global flag leaks into nvcc (CUDA) and breaks the build
  target_compile_options(whisper_scribe PRIVATE /utf-8)
endif()
target_include_directories(whisper_scribe PRIVATE ${miniaudio_SOURCE_DIR})
target_link_libraries(whisper_scribe PRIVATE
  whisper imgui_lib nfd Threads::Threads ${CMAKE_DL_LIBS})

if(WITH_PARAKEET)
  target_sources(whisper_scribe PRIVATE parakeet.cpp)
  target_compile_definitions(whisper_scribe PRIVATE WITH_PARAKEET=1)
  target_include_directories(whisper_scribe PRIVATE "${SHERPA_INCLUDE_DIR}")
  target_link_libraries(whisper_scribe PRIVATE "${SHERPA_C_API_LIB}")

  # runtime DLLs (sherpa-onnx-c-api.dll, onnxruntime.dll, ...) next to the exe
  file(GLOB_RECURSE _sherpa_runtime "${_sherpa_root}/*.dll")
  if(_sherpa_runtime)
    add_custom_command(TARGET whisper_scribe POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_sherpa_runtime} $<TARGET_FILE_DIR:whisper_scribe>)
  endif()
  if(NOT WIN32)
    get_filename_component(_sherpa_libdir "${SHERPA_C_API_LIB}" DIRECTORY)
    set_target_properties(whisper_scribe PROPERTIES BUILD_RPATH "${_sherpa_libdir}")
  endif()
endif()

message(STATUS "")
message(STATUS "WhisperScribe configuration")
message(STATUS "  GGML_CUDA:           ${GGML_CUDA}")
message(STATUS "  GGML_VULKAN:         ${GGML_VULKAN}")
message(STATUS "  CUDA architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Parakeet engine:     ${WITH_PARAKEET}")
message(STATUS "")
