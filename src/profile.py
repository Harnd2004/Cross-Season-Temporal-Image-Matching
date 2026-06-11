import json
from pathlib import Path
import pandas as pd


def save_latency_report(records, csv_path, summary_path):
    df = pd.DataFrame(records)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    n = len(df)
    ok_df = df[df["status"] == "ok"] if "status" in df.columns else df

    summary = {
        "num_pairs": int(n),
        "matcher": df["matcher"].iloc[0] if "matcher" in df.columns and n > 0 else "unknown",
        "avg_data_loading_ms":       float(df["data_loading_ms"].mean())       if n else 0.0,
        "avg_feature_matching_ms":   float(df["feature_matching_ms"].mean())   if n else 0.0,
        "avg_matrix_estimation_ms":  float(df["matrix_estimation_ms"].mean())  if n else 0.0,
        "avg_total_ms":              float(df["total_ms"].mean())               if n else 0.0,
        "avg_inlier_ratio":          float(df["inlier_ratio"].mean())           if n else 0.0,
        "max_inlier_ratio":          float(df["inlier_ratio"].max())            if n else 0.0,
        "pct_pairs_ok":              round(100.0 * len(ok_df) / n, 1)          if n else 0.0,
        "avg_inliers":               float(df["inliers"].mean())               if n else 0.0,
        "pairs_above_60pct_inliers": int((df["inlier_ratio"] >= 0.60).sum())   if n else 0,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return df, summary
