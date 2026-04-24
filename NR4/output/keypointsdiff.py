import json
import math
import pandas as pd
import matplotlib.pyplot as plt

KP_NAMES = [
    'Nase', 'L_Auge', 'R_Auge', 'L_Ohr', 'R_Ohr',
    'L_Schulter', 'R_Schulter', 'L_Ellbogen', 'R_Ellbogen',
    'L_Handgelenk', 'R_Handgelenk', 'L_Hüfte', 'R_Hüfte',
    'L_Knie', 'R_Knie', 'L_Knöchel', 'R_Knöchel'
]

dateien = {
    'lt-i': [
        'predictions/pr-lt-i-bild1.json',
        'predictions/pr-lt-i-bild2.json',
        'predictions/pr-lt-i-bild3.json'
    ],
    'lt-l': [
        'predictions/pr-lt-l-bild1.json',
        'predictions/pr-lt-l-bild2.json',
        'predictions/pr-lt-l-bild3.json'
    ],
    'lt-lm': [
        'predictions/pr-lt-lm-bild1.json',
        'predictions/pr-lt-lm-bild2.json',
        'predictions/pr-lt-lm-bild3.json'
    ],
    'pc-i': [
        'predictions/pr-pc-i-bild1.json',
        'predictions/pr-pc-i-bild2.json',
        'predictions/pr-pc-i-bild3.json'
    ],
    'pc-l': [
        'predictions/pr-pc-l-bild1.json',
        'predictions/pr-pc-l-bild2.json',
        'predictions/pr-pc-l-bild3.json'
    ],
    'pc-lm': [
        'predictions/pr-pc-lm-bild1.json',
        'predictions/pr-pc-lm-bild2.json',
        'predictions/pr-pc-lm-bild3.json'
    ]
}

def load_keypoints(path):
    with open(path) as f:
        data = json.load(f)[0]

    kpts = data["keypoints"]

    if isinstance(kpts[0][0], list):
        kpts = kpts[0]

    return kpts


# alle Daten laden
daten = {}
for setup in dateien:
    daten[setup] = [load_keypoints(p) for p in dateien[setup]]

rows = []

# Mittelwert immer nur innerhalb desselben Bildes berechnen
for bild in range(3):

    mean_pose = []

    for kp in range(17):
        xs = []
        ys = []

        for setup in daten:
            xs.append(daten[setup][bild][kp][0])
            ys.append(daten[setup][bild][kp][1])

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        mean_pose.append([mean_x, mean_y])

    # Abstände aller Setups zu diesem Bild-Mittelwert
    for setup in daten:
        for kp in range(17):
            x = daten[setup][bild][kp][0]
            y = daten[setup][bild][kp][1]

            mx = mean_pose[kp][0]
            my = mean_pose[kp][1]

            dist = math.sqrt((x - mx)**2 + (y - my)**2)

            rows.append({
                "setup": setup,
                "bild": bild + 1,
                "keypoint": KP_NAMES[kp],
                "distance": dist
            })

df = pd.DataFrame(rows)
df.to_csv("abweichungen.csv", index=False)  # alle Einzelwerte

# Boxplot pro Setup
plt.figure(figsize=(8, 5))
df.boxplot(column="distance", by="setup")
plt.title("")
plt.suptitle("")
plt.xlabel("Aufbau")
plt.ylabel("Abweichung [Pixel]")
plt.tight_layout()
plt.savefig("boxplot_setups.pdf", dpi=300)
plt.show()

# Balkendiagramm pro Keypoint
mean_per_keypoint = df.groupby("keypoint")["distance"].mean()
plt.figure(figsize=(10, 5))
mean_per_keypoint.plot(kind="bar")
#plt.title("Mittlere Abweichung pro Keypoint")
plt.xlabel("Keypoint")
plt.ylabel("Abweichung [Pixel]")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("barplot_keypoints.pdf", dpi=300)
plt.show()

print(df.groupby("setup")["distance"].mean().round(3))
print(mean_per_keypoint.round(3))