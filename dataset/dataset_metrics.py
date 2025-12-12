import pandas as pd
import os

def global_summary_table(csv_paths):
    """
    csv_paths: dict mapping sphere_count -> csv_path
    Example: {3:"metrics_3.csv", 4:"metrics_4.csv", ...}
    """
    rows = []

    for n, path in csv_paths.items():
        df = pd.read_csv(path)

        # Compute summary statistics
        mean_iou = df["IoU"].mean()
        median_iou = df["IoU"].median()
        std_iou = df["IoU"].std()

        mean_err = df["center_err"].mean()
        median_err = df["center_err"].median()

        failure_rate = (df["IoU"] < 0.2).mean()

        rows.append({
            "N_Spheres": n,
            "Mean_IoU": round(mean_iou, 4),
            "Median_IoU": round(median_iou, 4),
            "Std_IoU": round(std_iou, 4),
            "Mean_Center_Err": round(mean_err, 2),
            "Median_Center_Err": round(median_err, 2),
            "Failure_Rate": round(failure_rate * 100, 2)
        })

    summary_df = pd.DataFrame(rows)
    return summary_df

def per_color_performance(csv_paths):
    """
    Aggregates color stats across all datasets.
    Produces a table: Color vs Mean IoU, Median IoU, Center Error, Fail Rate.
    """
    df_all = []

    for n, path in csv_paths.items():
        df = pd.read_csv(path)
        df["n_spheres"] = n
        df_all.append(df)

    df_all = pd.concat(df_all, ignore_index=True)

    rows = []
    for color, group in df_all.groupby("color"):
        rows.append({
            "Color": color,
            "Mean IoU": round(group["IoU"].mean(), 4),
            "Median IoU": round(group["IoU"].median(), 4),
            "Mean Center Err": round(group["center_err"].mean(), 2),
            "Failure Rate (%)": round((group["IoU"] < 0.2).mean() * 100, 2),
            "Samples": len(group)
        })

    return pd.DataFrame(rows)

def failure_mode_table(csv_paths):
    df_all = []

    for n, path in csv_paths.items():
        df = pd.read_csv(path)
        df["n_spheres"] = n

        # classify failures
        df["Failure_Type"] = "Good"
        df.loc[df["IoU"] < 0.1, "Failure_Type"] = "Miss"
        df.loc[(df["IoU"] >= 0.1) & (df["IoU"] < 0.4), "Failure_Type"] = "Partial"
        df.loc[df["center_err"] > 50, "Failure_Type"] = "Wrong Object"

        df_all.append(df)

    df_all = pd.concat(df_all)

    # Aggregate counts
    table = df_all.groupby("Failure_Type").size().reset_index(name="Count")
    table["Percent"] = round(table["Count"] / table["Count"].sum() * 100, 2)

    return table


# ==== EXAMPLE USAGE ====
csvs = {
    3: "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_3/metrics_3.csv",
    4: "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_4/metrics_4.csv",
    5: "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_5/metrics_5.csv",
    6: "/home/jay/Language Aware Manipulation/dataset/dataset/spheres_6/metrics_6.csv",
}

print("/n/n/n-----------------------")
print("Global IoU summary")
table1 = global_summary_table(csvs)
print(table1)

print("/n/n/n-----------------------")
print("Per color performance")
table2 = per_color_performance(csvs)
print(table2)

print("/n/n/n-----------------------")
print("Failure analysis")
table3 = failure_mode_table(csvs)
print(table3)
