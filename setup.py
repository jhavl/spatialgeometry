from setuptools import setup, find_packages
from os import path
import os

here = path.abspath(path.dirname(__file__))

req = [
    'spatialmath-python'
]

collision_req = [
    'pybullet'
]

dev_req = [
    'roboticstoolbox-python',
    'swift',
    'pytest',
    'pytest-cov',
    'flake8',
    'pyyaml',
]

docs_req = [
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx-autorun',
]

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# def package_files(directory):
#     paths = []
#     for (pathhere, _, filenames) in os.walk(directory):
#         for filename in filenames:
#             paths.append(os.path.join('..', pathhere, filename))
#     return paths


# extra_files = package_files('swift/public')

setup(
    name='spatialgeometry',

    version='0.1.0',

    description='A Shape and Geometry Package',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/jhavl/spatialgeometry',

    author='Jesse Haviland',

    license='MIT',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    python_requires='>=3.6',

    keywords='python robotics robotics-toolbox kinematics dynamics' \
             ' motion-planning trajectory-generation jacobian hessian' \
             ' control simulation robot-manipulator mobile-robot',

    packages=find_packages(exclude=["tests", "examples"]),

    # package_data={'swift': extra_files},

    include_package_data=True,

    install_requires=req,

    extras_require={
        'collision': collision_req,
        'dev': dev_req,
        'docs': docs_req,
        'vpython': vp_req
    }
)
