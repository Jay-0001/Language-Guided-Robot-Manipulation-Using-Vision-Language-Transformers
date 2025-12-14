import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CONF_PATHS = {
    3: "W:\Binghamton\Semester 2\Robot Perception\Project\Implementations\Language-Aware-Robot-Manipulation\dataset\dataset\spheres_3\confusion_3.json",
    4: "W:\Binghamton\Semester 2\Robot Perception\Project\Implementations\Language-Aware-Robot-Manipulation\dataset\dataset\spheres_4\confusion_4.json",
    5: "W:\Binghamton\Semester 2\Robot Perception\Project\Implementations\Language-Aware-Robot-Manipulation\dataset\dataset\spheres_5\confusion_5.json",
    6: "W:\Binghamton\Semester 2\Robot Perception\Project\Implementations\Language-Aware-Robot-Manipulation\dataset\dataset\spheres_6\confusion_6.json",
}

SPHERE_PHRASES = [
    "red sphere",
    "green sphere",
    "blue sphere",
    "yellow sphere",
    "black sphere",
    "pink sphere",
]


def load_confusions():
    conf = {}
    for n, path in CONF_PATHS.items():
        with open(path, "r") as f:
            raw = json.load(f)
        parsed = {}
        for k, v in raw.items():
            gt, pred = eval(k)
            parsed.setdefault(gt, {"correct": 0, "wrong": 0})
            if pred == gt:
                parsed[gt]["correct"] += v
            else:
                parsed[gt]["wrong"] += v
        conf[n] = parsed
    return conf


def build_tables(conf):
    rows = []

    for n in conf:
        for color in SPHERE_PHRASES:
            if color not in conf[n]: continue
            correct = conf[n][color]["correct"]
            wrong = conf[n][color]["wrong"]
            total = correct + wrong
            acc = correct / total * 100 if total > 0 else 0

            rows.append({
                "n_spheres": n,
                "color": color,
                "correct": correct,
                "wrong": wrong,
                "accuracy": acc,
            })

    df = pd.DataFrame(rows)
    return df


def plot_accuracy(df):
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="n_spheres", y="accuracy", hue="color", marker="o")
    plt.title("Per-Color Accuracy vs Scene Complexity")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.show()


def plot_failure_breakdown(df):
    failure_summary = df.groupby("n_spheres")["wrong"].sum() / df.groupby("n_spheres")[["wrong","correct"]].sum().sum(axis=1)
    failure_summary.plot(kind="bar", figsize=(8,5), title="Wrong Object Rate by Sphere Count")
    plt.ylabel("Failure Rate")
    plt.show()


# === RUN ===
conf = load_confusions()
df = build_tables(conf)

print("\n=== Accuracy Table ===")
print(df)

print("\n=== Mean Accuracy Per Color ===")
print(df.groupby("color")["accuracy"].mean().sort_values(ascending=False))

plot_accuracy(df)
plot_failure_breakdown(df)
