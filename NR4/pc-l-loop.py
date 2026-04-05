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
import os
import csv

# Module registrieren
register_det_modules()
register_pose_modules(init_default_scope=False)

# ── Konfiguration ────────────────────────────────────────────────

POSE_CONFIG  = '/home/mci/projects/motioncapture-ba/NR4/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-t_8xb256-420e_coco-256x192.py'
POSE_WEIGHTS = '/home/mci/projects/motioncapture-ba/NR4/weights/rtmpose_tiny.pth'

DET_CONFIG   = '/home/mci/projects/motioncapture-ba/NR4/mmpose/demo/mmdetection_cfg/rtmdet_nano_320-8xb32_coco-person.py'
DET_WEIGHTS  = '/home/mci/projects/motioncapture-ba/NR4/weights/rtmdet_nano.pth'

VIDEO_PATH   = '/home/mci/projects/motioncapture-ba/NR4/input/Bild3.jpg'
OUTPUT_DIR   = '/home/mci/projects/motioncapture-ba/NR4/output'
CSV_PATH     = '/home/mci/projects/motioncapture-ba/NR4/output/ergebnisse.csv'

DEVICE = 'cuda:0' #if torch.cuda.is_available() else 'cpu'
RUNS = 30
CURRENT_INPUT = 'Bild3'
CODE_NAME     = 'lt-l'
# ────────────────────────────────────────────────────────────────

# Dateien/Pfade prüfen
if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Video nicht gefunden: {VIDEO_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Modelle einmal laden
print("Lade Modelle ...")
det_model  = init_detector(DET_CONFIG, DET_WEIGHTS, device=DEVICE)
pose_model = init_model(POSE_CONFIG, POSE_WEIGHTS, device=DEVICE)

print(f"Starte {RUNS} Durchläufe ...")

for run in range(1, RUNS + 1):
    print(f"\n── Run {run}/{RUNS} ──────────────────────")

    # Visualizer pro Run neu initialisieren
    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(pose_model.dataset_meta)

    # Video pro Run neu öffnen
    cap = cv2.VideoCapture(VIDEO_PATH)
    assert cap.isOpened(), f"Video nicht gefunden: {VIDEO_PATH}"

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = os.path.join(OUTPUT_DIR, f'{CURRENT_INPUT}')
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_video,
        (w, h)
    )

    latencies      = []
    det_latencies  = []
    pose_latencies = []
    vis_latencies  = []

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        # ── Detection ────────────────────────────────────────────
        det_start = time.perf_counter()

        with DefaultScope.overwrite_default_scope('mmdet'):
            det_result = inference_detector(det_model, frame)

        det_end = time.perf_counter()
        det_latency = det_end - det_start

        bboxes = det_result.pred_instances.bboxes.cpu().numpy()
        scores = det_result.pred_instances.scores.cpu().numpy()
        labels = det_result.pred_instances.labels.cpu().numpy()

        person_bboxes = bboxes[(labels == 0) & (scores > 0.3)]

        # ── Pose ─────────────────────────────────────────────────
        pose_start = time.perf_counter()

        with DefaultScope.overwrite_default_scope('mmpose'):
            if len(person_bboxes) > 0:
                pose_results = inference_topdown(pose_model, frame, person_bboxes)
            else:
                pose_results = []

        pose_end = time.perf_counter()
        pose_latency = pose_end - pose_start

        # ── Visualisierung ──────────────────────────────────────
        vis_frame = frame.copy()

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

        # ── Gesamtzeit ──────────────────────────────────────────
        frame_end = time.perf_counter()
        total_latency = frame_end - frame_start

        det_latencies.append(det_latency)
        pose_latencies.append(pose_latency)
        vis_latencies.append(vis_latency)
        latencies.append(total_latency)

        frame_count += 1
        writer.write(vis_frame)

        # Optional anzeigen
        cv2.imshow('Pose Estimation', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if not latencies:
        print("Keine Frames verarbeitet.")
        continue

    # ── Statistik pro Run ───────────────────────────────────────
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    avg_fps     = 1 / avg_latency if avg_latency > 0 else 0

    avg_det  = statistics.mean(det_latencies)
    avg_pose = statistics.mean(pose_latencies)
    avg_vis  = statistics.mean(vis_latencies)

    ##print(f"Frames verarbeitet:       {frame_count}")
    #print(f"Durchschnittliche Latenz: {avg_latency:.4f} Sekunden")
    #print(f"Minimale Latenz:          {min_latency:.4f} Sekunden")
    #print(f"Maximale Latenz:          {max_latency:.4f} Sekunden")
    #print(f"Standardabweichung:       {std_latency:.4f} Sekunden")
    #print(f"Durchschnittliche FPS:    {avg_fps:.2f}")
    #print(f"Detection Latenz:         {avg_det:.4f} Sekunden")
    #print(f"Pose Latenz:              {avg_pose:.4f} Sekunden")
    #print(f"Visualisierung:           {avg_vis:.4f} Sekunden")

    # ── CSV speichern ──────────────────────────────────────────
    neue_zeile = {
        'code':        CODE_NAME,
        'input':       CURRENT_INPUT,
        'run':         run,
        'frame_count': frame_count,
        'avg_latency': round(avg_latency, 4),
        'min_latency': round(min_latency, 4),
        'max_latency': round(max_latency, 4),
        'std_latency': round(std_latency, 4),
        'avg_fps':     round(avg_fps, 2),
        'avg_det':     round(avg_det, 4),
        'avg_pose':    round(avg_pose, 4),
        'avg_vis':     round(avg_vis, 4),
    }

    felder = list(neue_zeile.keys())
    datei_existiert = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as f:
        writer_csv = csv.DictWriter(f, fieldnames=felder)
        if not datei_existiert:
            writer_csv.writeheader()
        writer_csv.writerow(neue_zeile)

print(f"\nFertig. Alle {RUNS} Runs gespeichert in: {CSV_PATH}")