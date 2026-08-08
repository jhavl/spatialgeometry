cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Locate Python (interpreter needed to find nanobind cmake dir)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Locate nanobind via the installed Python package
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED HINTS "${NB_DIR}")

# ---------------------------------------------------------------------------
# scene — scene-graph transform propagation
# Eigen is vendored as header-only (Core module only -- see
# src/spatialgeometry/cpp-extension/Eigen) in src/spatialgeometry/cpp-extension/Eigen/
# scene_nb.cpp is the nanobind glue; math lives inline in that file.
#
# Skipped entirely under Pyodide/Emscripten: no compiler there worth
# targeting, and spatialgeometry.scene has a pure-Python fallback
# (scene.py) for exactly this case -- CPython's import machinery prefers
# a compiled extension module over a same-named .py file when both are
# present, so building neither is what makes the fallback take over.
# ---------------------------------------------------------------------------
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    nanobind_add_module(scene
        src/spatialgeometry/cpp-extension/scene_nb.cpp
    )
    target_include_directories(scene PRIVATE
        src/spatialgeometry/cpp-extension
    )

    install(TARGETS scene DESTINATION spatialgeometry)
endif()
