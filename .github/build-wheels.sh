#!/bin/bash
set -u -x

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w /io/wheelhouse/
    fi
}

cd ./io

# yum install gfortran libopenblas-dev liblapack-dev

# Compile wheels
# cp3[1,7-9][-,0] == cp37-cp37 to cp310-cp310
for PYBIN in /opt/python/cp3[1,7-9][-,0]*/bin; do
    "${PYBIN}/pip" install numpy==1.17.4
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    repair_wheel "$whl"
done

# Install packages and test
#for PYBIN in /opt/python/cp3[6-8]*/bin/; do
#    "${PYBIN}/pip" install roboticstoolbox-python[dev,collision,vpython] --no-index -f /io/wheelhouse
#    ("${PYBIN}/pytest")
#done

ls ./wheelhouse
