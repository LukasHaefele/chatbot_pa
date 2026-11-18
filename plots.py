
import json
import numpy as np
import matplotlib.pyplot as plt

# "test": Testset-Ergebnisse (hier wird der optimierte Bot bewertet)
# "dev": Dev-Ergebnisse (hier ist es bei allen Runs der unoptimierte Bot)
FOLDER = "output/evaluator/test"

RUNS = {
    "Baseline": "26-11-18_08-12.json",
    "Light":    "26-11-18_08-02.json",
    "Medium":   "26-11-18_07-46.json",
    "Heavy":    "26-11-18_07-16.json",
}

# Farbpalette: Heavy wird positiv hervorgehoben
COLORS = {
    "Baseline": "#A0A0A0",  # Grau
    "Light":    "#9ECAE1",  # Hellblau
    "Medium":   "#3182BD",  # Kräftiges Blau
    "Heavy":    "#238B45",  # Kräftiges Grün
}


def load_scores(filename):
    with open(f"{FOLDER}/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d["rag_metric"] for d in data}


# --- Ergebnisse laden ---
scores = {name: load_scores(file) for name, file in RUNS.items()}

# Nur Fragen verwenden, die in allen Runs vorkommen
ids = sorted(set.intersection(*(set(s) for s in scores.values())))

values = {
    name: [s[i] for i in ids]
    for name, s in scores.items()
}

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

# Übergeordnete Überschrift
fig.suptitle(
    "Vergleich der RAG-Performance über verschiedene Optimierungsstufen",
    fontsize=14,
    fontweight="bold"
)


# --- Links: Score pro Frage ---
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
ax1.set_ylabel("RAG-Score")
ax1.set_title("RAG-Score pro Frage")
ax1.legend(frameon=False)
ax1.grid(axis="y", alpha=0.25, linestyle="--")
ax1.set_axisbelow(True)


# --- Rechts: Durchschnitt pro Run ---
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
ax2.set_ylabel("Durchschnittlicher RAG-Score")
ax2.set_title("Durchschnittlicher RAG-Score pro Run")
ax2.grid(axis="y", alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)


# --- Layout und Export ---
plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "rag_metric_runs.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()
