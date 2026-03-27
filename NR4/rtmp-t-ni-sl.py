from mmengine.registry import DefaultScope
from mmdet.utils import register_all_modules as register_det_modules
from mmdet.apis import init_detector, inference_detector
from mmpose.utils import register_all_modules as register_pose_modules
from mmpose.apis import init_model, inference_topdown
from mmpose.registry import VISUALIZERS
import cv2
import time
import statistics
import numpy as np

# Module registrieren – mmdet zuerst, mmpose ohne Scope-Override (Werkzeugkasten)
register_det_modules()
register_pose_modules(init_default_scope=False)

# ── Konfiguration ────────────────────────────────────────────────
POSE_CONFIG  = '/mnt/c/Users/floyu/mmpose_project/NR4/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
POSE_WEIGHTS = '/mnt/c/Users/floyu/mmpose_project/NR4/weights/rtmpose_tiny.pth'

DET_CONFIG   = '/mnt/c/Users/floyu/mmpose_project/NR4/mmpose/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py'
DET_WEIGHTS  = '/mnt/c/Users/floyu/mmpose_project/NR4/weights/rtmdet_nano.pth'

VIDEO_PATH   = '/mnt/c/Users/floyu/mmpose_project/NR4/FloGolf.mp4'
DEVICE       = 'cpu'
# ────────────────────────────────────────────────────────────────

# Modelle laden
det_model  = init_detector(DET_CONFIG, DET_WEIGHTS, device=DEVICE)
pose_model = init_model(POSE_CONFIG, POSE_WEIGHTS, device=DEVICE)

# Visualizer initialisieren
visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
visualizer.set_dataset_meta(pose_model.dataset_meta)

# Video öffnen
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Video nicht gefunden: {VIDEO_PATH}"

# Output-Video vorbereiten
fps_video = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    '/mnt/c/Users/floyu/mmpose_project/NR4/output/result.mp4',
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps_video, (w, h)
)

latencies      = []
det_latencies  = []
pose_latencies = []
vis_latencies  = []

frame_count = 0
print("Benchmark gestartet ...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_start = time.perf_counter()

    # Personendetektion – Scope explizit auf mmdet setzen
    det_start = time.perf_counter()

    with DefaultScope.overwrite_default_scope('mmdet'):
        det_result = inference_detector(det_model, frame)

    det_end = time.perf_counter()
    det_latency = det_end - det_start

    bboxes = det_result.pred_instances.bboxes.cpu().numpy()
    scores = det_result.pred_instances.scores.cpu().numpy()
    labels = det_result.pred_instances.labels.cpu().numpy()

    person_bboxes = bboxes[(labels == 0) & (scores > 0.3)]

    # Pose-Schätzung – Scope explizit auf mmpose setzen
    pose_start = time.perf_counter()

    with DefaultScope.overwrite_default_scope('mmpose'):
        if len(person_bboxes) > 0:
            pose_results = inference_topdown(pose_model, frame, person_bboxes)
        else:
            pose_results = []

    pose_end = time.perf_counter()
    pose_latency = pose_end - pose_start

    # ── Visualisierung ───────────────────────────────────────────
    vis_frame = frame.copy()
    vis_latency = 0

    vis_start = time.perf_counter()

    if len(pose_results) > 0:
        merged_sample = pose_results[0]

        if len(pose_results) > 1:
            pred_instances = [res.pred_instances for res in pose_results]
            merged_sample.pred_instances = pred_instances[0].cat(pred_instances)

        visualizer.add_datasample(
            'result',
            vis_frame,
            data_sample=merged_sample,
            draw_gt=False,
            draw_heatmap=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            kpt_thr=0.3
        )

        vis_frame = visualizer.get_image()

    vis_end = time.perf_counter()
    vis_latency = vis_end - vis_start

    frame_end = time.perf_counter()
    total_latency = frame_end - frame_start

    det_latencies.append(det_latency)
    pose_latencies.append(pose_latency)
    vis_latencies.append(vis_latency)
    latencies.append(total_latency)

    frame_count += 1

    writer.write(vis_frame)

    cv2.imshow('Pose Estimation', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# ── Benchmark-Ausgabe ────────────────────────────────────────────
avg_latency = statistics.mean(latencies)
min_latency = min(latencies)
max_latency = max(latencies)
std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
avg_fps     = 1 / avg_latency

avg_det  = statistics.mean(det_latencies)
avg_pose = statistics.mean(pose_latencies)
avg_vis  = statistics.mean(vis_latencies)

print("\n========== Benchmark Ergebnis ==========")
print(f"Frames verarbeitet:       {frame_count}")
print(f"Durchschnittliche Latenz: {avg_latency:.4f} Sekunden")
print(f"Minimale Latenz:          {min_latency:.4f} Sekunden")
print(f"Maximale Latenz:          {max_latency:.4f} Sekunden")
print(f"Standardabweichung:       {std_latency:.4f} Sekunden")
print(f"\nDurchschnittliche FPS:    {avg_fps:.2f}")

print("\n--- Detailanalyse Pipeline ---")
print(f"Detection Latenz:         {avg_det:.4f} Sekunden")
print(f"Pose Latenz:              {avg_pose:.4f} Sekunden")
print(f"Visualisierung:           {avg_vis:.4f} Sekunden")
print("=======================================\n")