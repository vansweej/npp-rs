set -ex

#sudo apt install -y default-jre
#sudo apt install -y libcairo2

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-license-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-license-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-misc-headers-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-misc-headers-10-2_10.2.89-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvdisasm-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvdisasm-10-2_10.2.89-1_amd64.deb


wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nsight-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nsight-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvvp-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvvp-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvrtc-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvrtc-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvrtc-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvrtc-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cusolver-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cusolver-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cusolver-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cusolver-dev-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-libcublas-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-libcublas-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-libcublas-dev-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-libcublas-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cufft-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cufft-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cufft-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cufft-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-curand-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-curand-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-curand-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-curand-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cusparse-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cusparse-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cusparse-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cusparse-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-npp-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-npp-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-npp-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-npp-dev-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvml-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-nvml-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvml-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvml-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvjpeg-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvjpeg-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvjpeg-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvjpeg-dev-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nsight-compute-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-nsight-compute-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nsight-systems-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-nsight-systems-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvgraph-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvgraph-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvgraph-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvgraph-dev-10-2_10.2.89-1_amd64.deb




wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-gdb-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-gdb-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvprof-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvprof-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-sanitizer-api-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-sanitizer-api-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-memcheck-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-memcheck-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-driver-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-driver-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cudart-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cudart-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cudart-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cudart-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cupti-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cupti-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cupti-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cupti-dev-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvtx-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvtx-10-2_10.2.89-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvcc-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvcc-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-cuobjdump-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-cuobjdump-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvprune-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvprune-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-command-line-tools-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-command-line-tools-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-visual-tools-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-visual-tools-10-2_10.2.89-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-compiler-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-compiler-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-tools-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-tools-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-samples-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-samples-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-documentation-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-documentation-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-libraries-dev-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-libraries-dev-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-libraries-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-libraries-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-nvml-dev-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-nvml-dev-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-toolkit-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-toolkit-10-2_10.2.89-1_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-runtime-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-runtime-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-demo-suite-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-demo-suite-10-2_10.2.89-1_amd64.deb
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-10-2_10.2.89-1_amd64.deb
#sudo dpkg -i cuda-10-2_10.2.89-1_amd64.deb

sudo apt-get update
