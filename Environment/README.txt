1. Install Anaconda and create an environment.
2. Using pip, install torch in the environment.
    This study uses version 2.2.2+cu121 which can be installed by running pip install torch --index-url https://download.pytorch.org/whl/cu121.
3. Install CUDA Toolkit 12.2 from https://developer.nvidia.com/cuda-12-2-0-download-archive. 
4. Install cuDNN 8.9.7 from https://developer.nvidia.com/rdp/cudnn-archive and extract the files into the CUDA directory from the previous step, replacing the existing items if prompted.
    Using pip, install packages in requirements.txt in the environment.
5. Install optimum and apache-beam separately after the installation of the other packages has concluded because, otherwise, there will be issues with the resolution of dependency conflicts causing the terminal to hang.