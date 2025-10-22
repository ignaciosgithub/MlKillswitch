#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "GPT-2 Killswitch GPU Setup"
echo "=========================================="

echo "Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

echo "Installing essential tools..."
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    python3-pip \
    python3-dev

echo "Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt-get install -y nvidia-driver-535
    echo "NVIDIA drivers installed. Please reboot and run this script again."
    exit 0
else
    echo "NVIDIA drivers already installed:"
    nvidia-smi
fi

echo "Checking CUDA..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA Toolkit 12.1..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-12-1
    
    echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
else
    echo "CUDA already installed:"
    nvcc --version
fi

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Installing PyTorch with CUDA 12.1..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing Hugging Face transformers and dependencies..."
pip3 install \
    transformers \
    datasets \
    accelerate \
    bitsandbytes \
    wandb \
    tensorboard \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm

pip3 install \
    jupyterlab \
    ipywidgets \
    pandas \
    numpy

echo "Creating workspace..."
mkdir -p ~/killswitch_training
cd ~/killswitch_training

if [ ! -d "MlKillswitch" ]; then
    echo "Cloning MlKillswitch repository..."
    git clone https://github.com/ignaciosgithub/MlKillswitch.git
fi

cd MlKillswitch

mkdir -p checkpoints
mkdir -p logs
mkdir -p results
mkdir -p datasets

echo "Testing PyTorch GPU support..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. (Optional) Login to Weights & Biases: wandb login"
echo "2. Run training: python3 train_gpt2_killswitch_gpu.py"
echo ""
echo "Useful commands:"
echo "  - Monitor GPU: watch -n 1 nvidia-smi"
echo "  - View logs: tensorboard --logdir logs/"
echo "  - Resume training: python3 train_gpt2_killswitch_gpu.py --resume"
echo ""
