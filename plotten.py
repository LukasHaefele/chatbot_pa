
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Konfiguration ---
FOLDER = "output/evaluator/test"
RUN_FILE = "26-11-18_08-02.json"  # Gewünschten Run auswählen

METRICS = {
    "Points": "point_score",
    "Source": "source_score",
    "Answerable": "answerable_score",
    "Language": "language_score",
}

# Farbpalette im Stil des bisherigen Diagramms
COLORS = {
    "Points":     "#A0A0A0",  # Grau
    "Source":     "#9ECAE1",  # Hellblau
    "Answerable": "#3182BD",  # Blau
    "Language":   "#238B45",  # Grün
}


# --- Daten laden ---
with open(f"{FOLDER}/{RUN_FILE}", "r", encoding="utf-8") as f:
    data = json.load(f)

# Fragen-IDs und Scores extrahieren
ids = [d["id"] for d in data]

values = {
    name: [d[json_key] for d in data]
    for name, json_key in METRICS.items()
}

# Durchschnitt pro Metrik berechnen
means = {
    name: np.mean(v)
    for name, v in values.items()
}


# --- Diagramm erstellen ---
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(16, 7),
    gridspec_kw={"width_ratios": [3, 1]}
)

fig.suptitle(
    "Bewertung der Antwortqualität – DSPy_light_run",
    fontsize=14,
    fontweight="bold"
)


# --- Links: Scores pro Frage ---
x = np.arange(len(ids))
width = 0.2

for i, (name, v) in enumerate(values.items()):
    offset = (i - (len(values) - 1) / 2) * width

    ax1.bar(
        x + offset,
        v,
        width,
        label=name,
        color=COLORS[name],
        edgecolor="white",
        linewidth=0.5
    )

# Fragen-Labels
question_labels = [
    f"Q{i+1} (id={question_id})"
    for i, question_id in enumerate(ids)
]

ax1.set_xticks(x)
ax1.set_xticklabels(
    question_labels,
    rotation=45,
    ha="right"
)

ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Score")
ax1.set_title("Scores pro Frage")
ax1.legend(frameon=False)
ax1.grid(axis="y", alpha=0.25, linestyle="--")
ax1.set_axisbelow(True)


# --- Rechts: Durchschnitt pro Metrik ---
names = list(means.keys())
avg = list(means.values())

bars = ax2.bar(
    names,
    avg,
    color=[COLORS[name] for name in names],
    edgecolor="white",
    linewidth=0.5
)

ax2.bar_label(
    bars,
    fmt="%.3f",
    padding=3,
    fontsize=9,
    fontweight="bold"
)

ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Durchschnittlicher Score")
ax2.set_title("Durchschnitt pro Metrik")
ax2.grid(axis="y", alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)

# Labels lesbar darstellen
ax2.tick_params(axis="x", rotation=35)


# --- Layout und Export ---
plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "run_metric_overview.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
