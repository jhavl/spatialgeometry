from setuptools import setup, find_packages, Extension
from os import path
import os
import numpy

def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", pathhere, filename))
    return paths


extra_folders = [
    "spatialgeometry/core",
]

extra_files = []
for extra_folder in extra_folders:
    extra_files += package_files(extra_folder)

scene = Extension(
    name="spatialgeometry.scene",
    sources=["./spatialgeometry/core/scene.cpp"],
    include_dirs=["./spatialgeometry/core/", numpy.get_include()],
)


setup(
    package_data={"spatialgeometry": extra_files},
    ext_modules=[scene],
)
