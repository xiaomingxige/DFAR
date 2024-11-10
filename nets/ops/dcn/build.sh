#!/usr/bin/env bash

# You may need to modify the following paths before compiling.
CUDA_HOME=/usr/local/cuda-12.2 \
CUDNN_INCLUDE_DIR=/usr/local/cuda-12.2/include \
CUDNN_LIB_DIR=/usr/local/cuda-12.2/lib64 \

python setup.py build_ext --inplace

if [ -d "build" ]; then
    rm -r build
fi
