
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- Konfiguration ---
FOLDER = "output/evaluator/test"

RUNS = {
    "Run 1": "26-11-18_08-02.json",
    "Run 2": "26-11-18_07-16.json",
}

METRICS = {
    "Points": "point_score",
    "Source": "source_score",
    "Answerable": "answerable_score",
    "Language": "language_score",
}

# Eigene Farben pro Metrik
METRIC_COLORS = {
    "Points":     "#A0A0A0",  # Grau
    "Source":     "#9ECAE1",  # Hellblau
    "Answerable": "#3182BD",  # Dunkelblau
    "Language":   "#238B45",  # Grün
}

# Helligkeit zur Unterscheidung der Runs
RUN_ALPHA = {
    "Run 1": 0.55,
    "Run 2": 1.0,
}


def load_scores(filename):
    with open(f"{FOLDER}/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        d["id"]: {
            metric: d[json_key]
            for metric, json_key in METRICS.items()
        }
        for d in data
    }


# --- Daten laden ---
scores = {
    name: load_scores(filename)
    for name, filename in RUNS.items()
}

# Nur gemeinsame Fragen berücksichtigen
ids = sorted(
    set.intersection(*(set(s) for s in scores.values()))
)

values = {
    run: {
        metric: [scores[run][i][metric] for i in ids]
        for metric in METRICS
    }
    for run in RUNS
}

# Durchschnitt pro Run und Metrik
means = {
    run: {
        metric: np.mean(v)
        for metric, v in metrics.items()
    }
    for run, metrics in values.items()
}


# --- Diagramm erstellen ---
fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(20, 8),
    gridspec_kw={"width_ratios": [3, 1]}
)

fig.suptitle(
    "Vergleich der Antwortqualität zwischen zwei Runs",
    fontsize=14,
    fontweight="bold"
)


# --- Links: Scores pro Frage und Metrik ---
x = np.arange(len(ids))

n_metrics = len(METRICS)
n_runs = len(RUNS)

group_width = 0.8
metric_width = group_width / n_metrics
bar_width = metric_width / n_runs

for metric_idx, metric in enumerate(METRICS):
    for run_idx, run in enumerate(RUNS):

        offset = (
            -group_width / 2
            + metric_idx * metric_width
            + run_idx * bar_width
            + bar_width / 2
        )

        ax1.bar(
            x + offset,
            values[run][metric],
            bar_width,
            color=METRIC_COLORS[metric],
            edgecolor="white",
            linewidth=0.4,
            alpha=RUN_ALPHA[run]
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
ax1.set_title("Scores pro Frage und Bewertungskriterium")
ax1.grid(axis="y", alpha=0.25, linestyle="--")
ax1.set_axisbelow(True)


# --- Legende: Farben = Metriken ---
metric_legend = [
    Patch(
        facecolor=METRIC_COLORS[metric],
        label=metric
    )
    for metric in METRICS
]

legend1 = ax1.legend(
    handles=metric_legend,
    title="Bewertungskriterium",
    loc="upper left",
    frameon=False
)

ax1.add_artist(legend1)

# Zweite Legende: Helligkeit = Run
run_legend = [
    Patch(
        facecolor="#666666",
        alpha=RUN_ALPHA[run],
        label=run
    )
    for run in RUNS
]

ax1.legend(
    handles=run_legend,
    title="Run",
    loc="upper right",
    frameon=False
)


# --- Rechts: Durchschnitt pro Metrik ---
metrics = list(METRICS.keys())
x_avg = np.arange(len(metrics))
avg_width = 0.35

for i, run in enumerate(RUNS):

    avg = [means[run][metric] for metric in metrics]
    offset = (i - 0.5) * avg_width

    bars = ax2.bar(
        x_avg + offset,
        avg,
        avg_width,
        label=run,
        color=[METRIC_COLORS[m] for m in metrics],
        alpha=RUN_ALPHA[run],
        edgecolor="white",
        linewidth=0.5
    )

    ax2.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
        fontweight="bold"
    )

ax2.set_xticks(x_avg)
ax2.set_xticklabels(metrics, rotation=30, ha="right")

ax2.set_ylim(0, 1.1)
ax2.set_ylabel("Durchschnittlicher Score")
ax2.set_title("Durchschnitt pro Metrik")
ax2.grid(axis="y", alpha=0.25, linestyle="--")
ax2.set_axisbelow(True)
ax2.legend(frameon=False)


# --- Layout und Export ---
plt.tight_layout(rect=[0, 0, 1, 0.94])

plt.savefig(
    "run_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# --- Durchschnittswerte ausgeben ---
print("\nDurchschnittliche Scores:")

for metric in METRICS:
    print(f"\n{metric}:")
    for run in RUNS:
        print(f"  {run}: {means[run][metric]:.3f}")

print(f"\nVerglichene Fragen: {len(ids)}")
