# Tom Henehan - June 9, 2026
"""Score graph choices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize graph choices.")
    parser.add_argument("--graph-dir", type=Path, default=Path("graph_transformer_all_weeks"))
    parser.add_argument("--acceptable-yards-gap", type=float, default=1.0)
    parser.add_argument("--acceptable-yards-pct", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.graph_dir / "graph_receiver_opportunities.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path)
    group_cols = ["gameId", "playId"]
    best_idx = df.groupby(group_cols)["graph_risk_adjusted_expected_yards"].idxmax()
    actual = df.loc[df["was_actual_target"].eq(1)].copy()
    best = df.loc[ best_idx, group_cols+ ["nflId", "displayName", "graph_risk_adjusted_expected_yards", "graph_predicted_yards_if_completed", "graph_completion_probability", "graph_interception_probability"]].rename(columns={"nflId": "graph_best_target_nflId", "displayName": "graph_best_target_name", "graph_risk_adjusted_expected_yards": "graph_best_expected_yards", "graph_predicted_yards_if_completed": "graph_best_yards_if_completed", "graph_completion_probability": "graph_best_completion_probability", "graph_interception_probability": "graph_best_interception_probability" })
    actual = actual.rename(columns={"nflId": "actual_target_nflId", "displayName": "actual_target_name", "graph_risk_adjusted_expected_yards": "graph_actual_expected_yards", "graph_predicted_yards_if_completed": "graph_actual_yards_if_completed", "graph_completion_probability": "graph_actual_completion_probability", "graph_interception_probability": "graph_actual_interception_probability"})
    out = actual.merge(best, on=group_cols, how="left")
    out["graph_missed_expected_yards"] = (out["graph_best_expected_yards"] - out["graph_actual_expected_yards"])
    out["graph_best_choice"] = out["actual_target_nflId"].eq(out["graph_best_target_nflId"])
    out["graph_acceptable_gap"] = (args.acceptable_yards_pct * out["graph_best_expected_yards"].clip(lower=0)).clip(lower=args.acceptable_yards_gap)
    out["graph_acceptable_choice"] = out["graph_missed_expected_yards"].le(out["graph_acceptable_gap"])
    out_path = args.graph_dir / "graph_play_decision_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote graph decision summary to {out_path}")


if __name__ == "__main__":
    main()
