
set -ex

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-toolkit-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-toolkit-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-runtime-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-runtime-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-demo-suite-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-demo-suite-10-2_10.2.89-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-10-2_10.2.89-1_amd64.deb
sudo dpkg -i cuda-10-2_10.2.89-1_amd64.deb
sudo apt-get update
