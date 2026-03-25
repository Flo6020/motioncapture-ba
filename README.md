# motioncapture-ba

# Setup:

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


# Zusätzliche Libraries installieren (Fix für OpenCV GUI / Qt xcb Fehler)

# Problem:
# Bei Nutzung von show=True kam es zu folgendem Fehler:
# qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
#
# Ursache:
# OpenCV nutzt unter Linux ein Qt-basiertes GUI-Backend.
# In WSL fehlen einige benötigte X11- und OpenGL-Runtimebibliotheken standardmäßig.  --> nachinstallieren
sudo apt update

sudo apt install -y \
libgl1 \
libglx-mesa0 \
libxcb-xinerama0 \
libxkbcommon-x11-0 \
libxcb1 \
libx11-xcb1 \
libxcb-render0 \
libxcb-shape0 \
libxcb-xfixes0 \
libxcb-randr0 \
libxcb-shm0 \
libxcb-icccm4 \
libxcb-keysyms1 \
libxcb-image0 \
libxcb-util1 \
libxcb-cursor0 \
libxcb-xkb1 \
libxcb-sync1 \
libxrender1 \
libfontconfig1 \
libfreetype6 \
libxext6 \
libsm6 \
libice6

# Problem:
# erzeugte MP4-Dateien konnten zunächst nicht abgespielt werden
sudo apt install ffmpeg

# Test der Video-Wiedergabe
ffplay output/visualizations/<video>.mp4
