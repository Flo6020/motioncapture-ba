import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# COCO-17 Skeleton
KP_NAMES = [
    'Nase', 'L_Auge', 'R_Auge', 'L_Ohr', 'R_Ohr',
    'L_Schulter', 'R_Schulter', 'L_Ellbogen', 'R_Ellbogen',
    'L_Handgelenk', 'R_Handgelenk', 'L_Hüfte', 'R_Hüfte',
    'L_Knie', 'R_Knie', 'L_Knöchel', 'R_Knöchel',
]

SKELETON = [
    (0,1),(0,2),(1,3),(2,4),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Daten laden 
def load(path):
    with open(path) as f:
        p = json.load(f)[0]
    kpts = p['keypoints']
    if isinstance(kpts[0][0], list):  # lt-l/pc-l haben [[...]]
        kpts = kpts[0]
    return kpts

# Alle 12 Dateien laden
d = {
    'lt-i': [
        load('predictions/pr-lt-i-bild1.json'),
        load('predictions/pr-lt-i-bild2.json'),
        load('predictions/pr-lt-i-bild3.json'),
    ],
    'lt-l': [
        load('predictions/pr-lt-l-bild1.json'),
        load('predictions/pr-lt-l-bild2.json'),
        load('predictions/pr-lt-l-bild3.json'),
    ],
    'pc-i': [
        load('predictions/pr-pc-i-bild1.json'),
        load('predictions/pr-pc-i-bild2.json'),
        load('predictions/pr-pc-i-bild3.json'),
    ],
    'pc-l': [
        load('predictions/pr-pc-l-bild1.json'),
        load('predictions/pr-pc-l-bild2.json'),
        load('predictions/pr-pc-l-bild3.json'),
    ],
}

# Farben & Marker
styles = {
    'lt-i': ('#E91E63', 'o', 'lt-i'),
    'lt-l': ('#2196F3', 's', 'lt-l'),
    'pc-i': ('#4CAF50', '^', 'pc-i'),
    'pc-l': ('#FF9800', 'D', 'pc-l'),
}

# 3 Plots erstellen
for bild in range(3):
    fig, ax = plt.subplots(figsize=(8, 12))
    #ax.set_title(f'Keypoint-Overlay — Bild {bild+1}', fontsize=14, fontweight='bold')

    for setup in ['lt-i', 'lt-l', 'pc-i', 'pc-l']:
        color, marker, label = styles[setup]
        kpts = d[setup][bild]
        xs = [k[0] for k in kpts]
        ys = [k[1] for k in kpts]

        for a, b in SKELETON:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, alpha=0.25, linewidth=1.8)

        ax.scatter(xs, ys, c=color, marker=marker, s=80, alpha=0.85,
                   edgecolors='white', linewidth=0.7, label=label, zorder=5)

    # Keypoint-Beschriftung (nur einmal)
    kpts0 = d['lt-i'][bild]
    for j in range(17):
        ax.annotate(KP_NAMES[j], (kpts0[j][0], kpts0[j][1]),
                    textcoords='offset points', xytext=(10, 4), fontsize=7, alpha=0.5)

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.15)
    ax.set_xlabel('X (Pixel)')
    ax.set_ylabel('Y (Pixel)')
    ax.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'visualizations/scatter_bild{bild+1}.pdf', dpi=200)
    plt.close()
    print(f'-> scatter_bild{bild+1}.pdf')

print('Fertig!')