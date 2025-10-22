Copy paste this for custom python and cuda using miniconda

    # Set Python version
    !wget -O mini.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh
    !chmod +x mini.sh
    !bash ./mini.sh -b -f -p /usr/local
    !conda install -q -y python=3.7
    import sys
    sys.path.append('/usr/local/lib/python3.7/site-packages')
    !python --version  # Should say Python 3.7.x

    # Set CUDA download URL 
    !wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    !chmod +x cuda_11.8.0_520.61.05_linux.run
    !./cuda_11.8.0_520.61.05_linux.run --silent --toolkit --no-drm --no-man-page
    import os
    os.environ['PATH'] += ':/usr/local/cuda-11.8/bin'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.8/lib64:/usr/lib64-nvidia'
    !nvcc --version  # Should show CUDA 11.8

    # Example torch* modules installation
    !pip uninstall torch torchvision torchaudio -y
    !pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

    # Verify proper installation
    import torch
    print(torch.cuda.is_available())  # Should be True
    print(torch.version.cuda)        # Should be 11.3 (from PyTorch)
    print(torch.cuda.get_device_name(0))  # Should show GPU