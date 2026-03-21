# motioncapture-ba

Setup:

# WSL installieren (PowerShell als Admin)
wsl --install

# Ubuntu starten und im Linux-Dateisystem arbeiten
cd ~
mkdir mmpose_project #neuer Ordner
cd mmpose_project

# Miniconda in WSL installieren
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Conda aktivieren
source ~/.bashrc

# falls conda nicht erkannt wird
source ~/miniconda3/bin/activate
conda init bash
source ~/.bashrc

# Conda Terms akzeptieren (wichtig bei neuen Versionen)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Environment erstellen
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# PyTorch installieren (CPU)
conda install pytorch torchvision cpuonly -c pytorch

# wichtige zusätzliche Dependencies
pip install fsspec
pip install -U openmim
pip install mmengine

# System-Compiler installieren (notwendig für mmcv)
sudo apt update
sudo apt install -y build-essential
g++ --version

# OpenMMLab Pakete installieren
mim install "mmcv==2.1.0"   # dauert ca. 10–15 Minuten
mim install "mmdet==3.2.0"
mim install "mmpose>=1.1.0"