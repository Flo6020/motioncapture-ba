from mmpose.apis import MMPoseInferencer
import time
import statistics
import csv
import os

RUNS = 30
CURRENT_INPUT = 'input/Bild3.jpg'  
CODE_NAME     = 'pc-i' #oder lt-i             
CSV_PATH      = '/home/mci/projects/motioncapture-ba/NR4/output/ergebnisse.csv'

# Inferencer einmal laden, außerhalb der Schleife
inferencer = MMPoseInferencer(
    pose2d='rtmpose-t_8xb256-420e_coco-256x192'
)

print(f"Starte {RUNS} Durchläufe ...")

for run in range(1, RUNS + 1):
    print(f"\n── Run {run}/{RUNS} ──────────────────────")

    results = inferencer(
        CURRENT_INPUT,
        show=False,       
        out_dir='output'
    )

    latencies   = []
    frame_count = 0

    while True:
        start = time.perf_counter()
        try:
            next(results)
        except StopIteration:
            break
        end = time.perf_counter()
        latencies.append(end - start)
        frame_count += 1

    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0
    avg_fps     = 1 / avg_latency

    print(f"Frames: {frame_count} | Avg: {avg_latency:.4f}s | FPS: {avg_fps:.2f}")

    # CSV speichern
    neue_zeile = {
        'code':        CODE_NAME,
        'input':       os.path.basename(CURRENT_INPUT),
        'run':         run,
        'frame_count': frame_count,
        'avg_latency': round(avg_latency, 4),
        'min_latency': round(min_latency, 4),
        'max_latency': round(max_latency, 4),
        'std_latency': round(std_latency, 4),
        'avg_fps':     round(avg_fps, 2),
    }

    felder          = list(neue_zeile.keys())
    datei_existiert = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=felder)
        if not datei_existiert:
            writer.writeheader()
        writer.writerow(neue_zeile)

print(f"\nFertig. Alle {RUNS} Runs gespeichert in: {CSV_PATH}")
