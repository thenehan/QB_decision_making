# Tom Henehan - June 9, 2026
"""Make project plots."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make project visuals.")
    parser.add_argument("--output-dir", type=Path, default=Path("expected_yards_output_all_weeks"))
    return parser.parse_args()
def save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    ax = df.plot(kind="bar", x=x, y=y, legend=False, figsize=(9, 5))
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    ax = df.plot(kind="scatter", x=x, y=y, figsize=(7, 5), alpha=0.5)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
def save_calibration(df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    for split, part in df.dropna(subset=["avg_predicted", "actual_rate"]).groupby("split"):
        ax.plot(part["avg_predicted"], part["actual_rate"], marker="o", label=split)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
def main() -> None:
    args = parse_args()
    analysis_dir = args.output_dir / "analysis"
    fig_dir = analysis_dir / "figures"

    qb_path = analysis_dir/"qb_summary_all.csv"
    play_path = analysis_dir/"enriched_play_decision_summary.csv"
    error_path = analysis_dir/"model_error_comparison.csv"
    comp_cal_path = analysis_dir/"completion_calibration.csv"
    int_cal_path = analysis_dir/"interception_calibration.csv"

    if qb_path.exists():
        qbs = pd.read_csv(qb_path)
        qbs = qbs.sort_values("missed_xyards").head(15)
        save_bar(qbs, "passer_name", "missed_xyards", fig_dir/"qb_missed_yards.png", "QB Missed Expected Yards")

        if "acceptable_choice_rate_p50" in qbs.columns:
            top = qbs.sort_values("acceptable_choice_rate_p50", ascending=False).head(15)
            save_bar(top, "passer_name", "acceptable_choice_rate_p50",fig_dir/"qb_acceptable_rate.png", "QB Acceptable Choice Rate")

    if play_path.exists():
        plays = pd.read_csv(play_path)
        save_scatter(plays, "actual_target_expected_yards_p50","best_available_expected_yards_p50", fig_dir / "actual_vs_best_value.png", "Actual Target vs Best Available",)
        if "timing_loss_expected_yards_p50" in plays.columns:
            timing = (plays.groupby("passer_name", dropna=False).agg(throws=("playId", "size"), timing_loss=("timing_loss_expected_yards_p50", "mean")).reset_index().loc[lambda df: df["throws"].ge(1)].sort_values("timing_loss").head(15))
            save_bar(timing,"passer_name","timing_loss", fig_dir / "qb_timing_loss.png", "QB Timing Loss")

    if error_path.exists():
        errors = pd.read_csv(error_path)
        import matplotlib.pyplot as plt

        pivot = errors.pivot(index="target", columns="model", values="mae")
        ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
        ax.set_title("Model Error")
        ax.set_ylabel("MAE")
        ax.grid(axis="y",alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir /"model_error.png", dpi=180)
        plt.close()

    if comp_cal_path.exists():
        save_calibration(pd.read_csv(comp_cal_path), fig_dir /"completion_calibration.png", "Completion Calibration")

    if int_cal_path.exists():
        save_calibration(pd.read_csv(int_cal_path), fig_dir/"interception_calibration.png", "Interception Calibration",)
    print(f"Wrote visuals to {fig_dir}")


if __name__ == "__main__":
    main()
