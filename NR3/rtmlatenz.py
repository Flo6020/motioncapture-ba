from mmpose.apis import MMPoseInferencer
import time
import statistics

inferencer = MMPoseInferencer(
    pose2d='rtmpose-t_8xb256-420e_coco-256x192'
)

results = inferencer(
    'FloGolf.mp4',
    show=True,
    out_dir='output'
)

latencies = []
frame_count = 0

print("Benchmark gestartet ...")

while True:

    start = time.perf_counter()

    try:
        next(results)
    except StopIteration:
        break

    end = time.perf_counter()

    latency = end - start
    latencies.append(latency)

    frame_count += 1

print("Video fertig verarbeitet")

avg_latency = statistics.mean(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

if len(latencies) > 1:
    std_latency = statistics.stdev(latencies)
else:
    std_latency = 0

avg_fps = 1 / avg_latency


print("\n========== Benchmark Ergebnis ==========")
print(f"Frames verarbeitet: {frame_count}")
print(f"Durchschnittliche Latenz: {avg_latency:.4f} Sekunden")
print(f"Minimale Latenz:          {min_latency:.4f} Sekunden")
print(f"Maximale Latenz:          {max_latency:.4f} Sekunden")
print(f"Standardabweichung:       {std_latency:.4f} Sekunden")
print(f"\nDurchschnittliche FPS:    {avg_fps:.2f}")
print("=======================================\n")