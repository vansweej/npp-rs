# npp-rs
![main workflow](https://github.com/vansweej/npp-rs/actions/workflows/build.yml/badge.svg)

This repository provides rust bindings to the Nvidia NPP libraries. 
Currently a subset of image processing is developed to be used in Neural Network processing. This crate is developed for CUDA 10.2 but 11.x support will be added later.
This crate is supported on Linux and Windows 10. For building on Linux use the CUDA_INSTALL_DIR environment variable to point to the root of your CUDA installation. For Windows 10, check if the CUDA_PATH environment variable has been set by the installer, otherwise add it manually pointing to the root of your CUDA installation.
