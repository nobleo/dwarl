# Provides the imported target onnxruntime::onnxruntime.
#   -DONNXRUNTIME_ROOT=/path  use an existing install (e.g. a JetPack GPU/TensorRT build) with include/ + lib/
#   otherwise the official prebuilt CPU release for this architecture is downloaded once at configure time.
set(ONNXRUNTIME_VERSION "1.20.1" CACHE STRING "ONNX Runtime release to download")
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "Existing ONNX Runtime install (include/ + lib/)")

if(NOT ONNXRUNTIME_ROOT)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_arch "aarch64")
  else()
    set(_ort_arch "x64")
  endif()
  set(_ort_name "onnxruntime-linux-${_ort_arch}-${ONNXRUNTIME_VERSION}")
  set(_ort_dir "${CMAKE_BINARY_DIR}/_deps/${_ort_name}")
  if(NOT EXISTS "${_ort_dir}/lib")
    set(_url "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${_ort_name}.tgz")
    message(STATUS "Downloading ONNX Runtime: ${_url}")
    file(DOWNLOAD "${_url}" "${CMAKE_BINARY_DIR}/_deps/${_ort_name}.tgz" STATUS _st SHOW_PROGRESS)
    list(GET _st 0 _code)
    if(NOT _code EQUAL 0)
      message(FATAL_ERROR "ONNX Runtime download failed (${_st}); set -DONNXRUNTIME_ROOT=<install>")
    endif()
    file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/_deps/${_ort_name}.tgz" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
  endif()
  set(ONNXRUNTIME_ROOT "${_ort_dir}")
  set(ONNXRUNTIME_BUNDLED ON)
endif()

find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
        "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session" NO_DEFAULT_PATH REQUIRED)
add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")

# An existing install (e.g. pixi's $CONDA_PREFIX, whose lib/ CMake treats as implicit and drops from the
# rpath) must still be found at run time by the plugin and the gtest binaries.
if(NOT ONNXRUNTIME_BUNDLED)
  get_filename_component(_ort_libdir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  list(APPEND CMAKE_BUILD_RPATH "${_ort_libdir}")
  set(ONNXRUNTIME_RPATH "${_ort_libdir}")
endif()
