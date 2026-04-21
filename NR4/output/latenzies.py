import pandas as pd
import matplotlib.pyplot as plt

# Pfade zu den vier CSV-Dateien
dateien = {
    "pc-i": "latenz-pc-i.csv",
    "pc-l": "latenz-pc-l.csv",
    "lt-i": "latenz-lt-i.csv",
    "lt-l": "latenz-lt-l.csv"
}

# CSV-Dateien einlesen
alle_dfs = []

for aufbau, pfad in dateien.items():
    df = pd.read_csv(pfad)

    # Aufbau aus dem Dateinamen übernehmen
    df["aufbau"] = aufbau

    alle_dfs.append(df)

# Alle Daten zu einer großen Tabelle zusammenfügen
gesamt_df = pd.concat(alle_dfs, ignore_index=True)

# Sekunden --> Millisekunden
spalten_sekunden = [
    "avg_latency", "min_latency", "max_latency", "std_latency",
    "avg_det", "avg_pose", "avg_vis"
]
for spalte in spalten_sekunden:
    if spalte in gesamt_df.columns:
        gesamt_df[spalte] = gesamt_df[spalte] * 1000

# Mittelwerte der Hauptlatenzen von setups
tabelle_mean = (
    gesamt_df.groupby("aufbau")[["avg_latency", "min_latency", "max_latency", "std_latency", "avg_fps"]]
    .mean()
    .round(4)
    .reset_index()
)

# Tabelle speichern
tabelle_mean.to_csv("mittelwerte_pro_aufbau.csv", index=False)
print("\nDatei gespeichert als: mittelwerte_pro_aufbau.csv")
print(tabelle_mean)


# Nur det/pose/vis zeilen
spalten_teil = ["avg_det", "avg_pose", "avg_vis"]

df_teil = gesamt_df.dropna(subset=spalten_teil)

# Mittelwerte det/pose/vis berechnen
tabelle_teil_mean = (
    df_teil.groupby("aufbau")[spalten_teil]
    .mean()
    .round(4)
    .reset_index()
)

# Tabelle speichern
tabelle_teil_mean.to_csv("mittelwerte_teil_latenzen.csv", index=False)
print("\nDatei gespeichert als: mittelwerte_teil_latenzen.csv")
print(tabelle_teil_mean)


# Diagramm 1: Vergleich der mittleren avg_latency aller 4 Aufbauten
plt.figure(figsize=(8, 5))
plt.bar(tabelle_mean["aufbau"], tabelle_mean["avg_latency"])
#plt.title("Mittlere Latenz pro Aufbau")
plt.xlabel("Aufbau")
plt.ylabel("Mittlere avg_latency [ms]")
plt.tight_layout()
plt.savefig("diagramm_avg_latency.pdf", dpi=300)
plt.show()

# Diagramm 2: Vergleich von avg_det, avg_pose, avg_vis
tabelle_teil_plot = tabelle_teil_mean.set_index("aufbau")

plt.figure(figsize=(8, 5))
tabelle_teil_plot.plot(kind="bar")
#plt.title("Mittlere Teil-Latenzen pro Aufbau")
plt.xlabel("Aufbau")
plt.ylabel("Zeit [ms]")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("diagramm_teil_latenzen.pdf", dpi=300)
plt.show()

# Diagramm 3: Boxplot der avg_latency für alle 4 Aufbauten in einem Diagramm

plt.figure(figsize=(8, 5))

gesamt_df.boxplot(
    column="avg_latency",
    by="aufbau",
    grid=False
)

plt.title("Verteilung der avg_latency pro Aufbau")
plt.suptitle("")  # entfernt automatischen pandas-Titel
plt.xlabel("Aufbau")
plt.ylabel("avg_latency [ms]")

plt.tight_layout()
plt.savefig("diagramm_boxplot_avg_latency.png", dpi=300)
plt.show()

'''
# Diagramm 3: Zwei Boxplots nebeneinander – nur Video-Inputs
# Links: schnelle Setups (pc-i, pc-l, lt-l), Rechts: langsamer Aufbau (lt-i)
nur_videos = gesamt_df[gesamt_df["input"].str.contains("Video", case=False, na=False)]
 
schnelle_aufbauten = ["pc-i", "pc-l", "lt-l"]
langsame_aufbauten = ["lt-i"]
 
daten_schnell = [
    nur_videos.loc[nur_videos["aufbau"] == a, "avg_latency"].values
    for a in schnelle_aufbauten
]
daten_langsam = [
    nur_videos.loc[nur_videos["aufbau"] == a, "avg_latency"].values
    for a in langsame_aufbauten
]
 
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 5),
    gridspec_kw={'width_ratios': [3, 1]}
)

# Linker Plot: schnelle Setups
ax1.boxplot(daten_schnell, labels=schnelle_aufbauten,
            showmeans=True, meanline=True)
max_schnell = max(d.max() for d in daten_schnell if len(d) > 0)
ax1.set_ylim(top=max_schnell * 1.25)
for i, daten in enumerate(daten_schnell, start=1):
    if len(daten) == 0:
        continue
    m = daten.mean()
    s = daten.std(ddof=1) if len(daten) > 1 else 0
    ax1.text(i, daten.max() + max_schnell * 0.04,
             f'µ={m:.2f}\nσ={s:.2f}\nn={len(daten)}',
             ha='center', va='bottom', fontsize=9)
ax1.set_title("Schnelle Aufbauten")
ax1.set_xlabel("Aufbau")
ax1.set_ylabel("avg_latency [ms]")
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
 
# Rechter Plot: lt-i
ax2.boxplot(daten_langsam, labels=langsame_aufbauten,
            showmeans=True, meanline=True)
max_langsam = max(d.max() for d in daten_langsam if len(d) > 0)
ax2.set_ylim(top=max_langsam * 1.25)
for i, daten in enumerate(daten_langsam, start=1):
    if len(daten) == 0:
        continue
    m = daten.mean()
    s = daten.std(ddof=1) if len(daten) > 1 else 0
    ax2.text(i, daten.max() + max_langsam * 0.04,
             f'µ={m:.2f}\nσ={s:.2f}\nn={len(daten)}',
             ha='center', va='bottom', fontsize=9)
ax2.set_title("Langsamer Aufbau")
ax2.set_xlabel("Aufbau")
ax2.set_ylabel("avg_latency [ms]")
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
 
fig.suptitle("Verteilung der mittleren Latenz pro Aufbau (nur Videos)",
             fontsize=12)
plt.tight_layout()
plt.savefig("diagramm_boxplot_latenz.png", dpi=300)
plt.show()'''