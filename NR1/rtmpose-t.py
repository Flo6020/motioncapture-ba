# Diese API kapselt die komplette Pipeline bestehend aus:
# - Personenerkennung (Bounding Box Detection)
# - 2D-Keypoint-Schätzung
# in einer einzigen einfach nutzbaren Klasse.
from mmpose.apis import MMPoseInferencer


# Initialisierung des Inferenzobjekts.
# 'rtmpose-t_8xb256-420e_coco-256x192' bedeutet:
# rtmpose-t      → Tiny-Version von RTMPose (optimiert für geringe Latenz)
# 8xb256         → Trainings-Batchstruktur (irrelevant für Inferenz)
# 420e           → 420 Trainings-Epochen während Modelltraining
# coco           → Trainingsdatensatz (COCO Keypoint Dataset)
# 256x192        → Eingabebildgröße des Netzwerks
#
# Beim ersten Start lädt MMPose automatisch die passenden Gewichte herunter.
inferencer = MMPoseInferencer(
    pose2d='rtmpose-t_8xb256-420e_coco-256x192'
)

# Die Funktion gibt einen Generator zurück, der Frame für Frame Ergebnisse liefert.
results = inferencer(
    'VNpose.jpg',
    show=True, #True→ aktiviert ein Fenster mit visualisierten Keypoints und Bounding Boxes
    out_dir='output' # → speichert zusätzlich
)

# Bei Videos würde diese Schleife alle Frames nacheinander verarbeiten.
for _ in results:
    pass

print("fertig")