# motioncapture-ba

Bachelorarbeit: Latenzoptimierte 2D-Pose-Schätzung auf Basis von MMPose  
für den möglichen Einsatz in der Rehabilitationsunterstützung und im Reha-Sport.

Das System erkennt menschliche Körperposen in Echtzeit aus Kamera- oder
Videoeingaben und visualisiert die detektierten Keypoints als Live-Overlay.
Ziel ist eine möglichst geringe Ende-zu-Ende-Latenz, um unmittelbares
Bewegungsfeedback zu ermöglichen.

---

## Inhalt

- [Zielsetzung](#zielsetzung)
- [Installation](#installation)

---

## Zielsetzung

Konventionelle Motion-Capture-Systeme sind präzise, aber teuer und an feste
Infrastruktur gebunden. Dieses Projekt untersucht, ob eine markerlose,
kamerabasierte 2D-Pose-Schätzung für Feedback-Anwendungen im Reha-Sport
ausreichend geringe Latenzen erzielen kann.

Das System basiert auf **MMPose** (OpenMMLab) und ist für zwei Szenarien
ausgelegt:

- **CPU-Betrieb** – Laptop ohne dedizierte GPU 
- **GPU-Betrieb** – Arbeitsplatzrechner mit CUDA 

---

## Installation

Als Ausgangspunkt dient die offizielle MMPose-Installationsanleitung:  
[mmpose.readthedocs.io](https://mmpose.readthedocs.io/en/latest/installation.html)

### Kurzinstallation 

```bash
# Conda-Umgebung erstellen
conda create --name openmmlab python=3.10 -y
conda activate openmmlab

# PyTorch mit GPU
pip install torch==2.1.2 torchvision \
  --index-url [download.pytorch.org](https://download.pytorch.org/whl/cu118)

# PyTorch mit CPU
conda install pytorch torchvision cpuonly -c pytorch

# Buildtools (einmalig in WSL)
sudo apt install -y build-essential

# Python-Abhängigkeiten
pip install fsspec
pip install -U openmim
pip install mmengine

# MMCV für GPU als vorkompiliertes CUDA-Wheel 
pip install mmcv==2.1.0 \
  -f [download.openmmlab.com](https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html)

# MMCV für GPU
mim install "mmcv==2.1.0"

# OpenMMLab-Pakete
mim install "mmdet==3.2.0"
mim install "mmpose>=1.1.0"

# Versionsanpassungen (Kompatibilität)
pip install "numpy<2"
pip install "opencv-python<4.13"
