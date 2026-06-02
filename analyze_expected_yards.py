"""Make summary tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path



def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize QB decision quality from expected receiver yards outputs."
    )
    parser.add_argument("--output-dir",type=Path,default=Path("expected_yards_output_all_weeks"))
    parser.add_argument("--data-dir",type=Path,default=Path("."))
    parser.add_argument("--min-throws", type=int, default=1)
    return parser.parse_args()


def read_csv(path: Path, **kwargs)->pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find{path}")
    return pd.read_csv(path, **kwargs)


def enrich_play_summary(summary: pd.DataFrame,data_dir: Path) -> pd.DataFrame:
    plays = read_csv(
        data_dir / "plays.csv",
        usecols=[
            "gameId","playId","possessionTeam","defensiveTeam","down","yardsToGo","offenseFormation",
            "receiverAlignment","passResult","timeToThrow","expectedPointsAdded","yardsGained","pff_passCoverage",
            "pff_manZone",
        ],
    )
    player_play = read_csv(
        data_dir/"player_play.csv",
        usecols=["gameId","playId","nflId","hadDropback"],
    )
    players = read_csv(data_dir/"players.csv", usecols=["nflId","displayName", "position"])
    passers = (player_play.loc[player_play["hadDropback"].eq(1), ["gameId", "playId", "nflId"]].drop_duplicates(["gameId", "playId"]).merge(players, on="nflId", how="left").rename(columns={"nflId": "passer_nflId","displayName": "passer_name","position": "passer_position",}))
    tracking_passers = infer_passers_from_tracking(summary,plays, players, data_dir)

    enriched = summary.merge(plays, on=["gameId","playId"],how="left")
    enriched = enriched.merge(passers, on=["gameId","playId"],how="left")
    enriched = enriched.merge(tracking_passers, on=["gameId","playId"],how="left")
    enriched["passer_nflId"]=enriched["passer_nflId"].fillna(enriched["tracking_passer_nflId"])
    enriched["passer_name"] = enriched["passer_name"].fillna(enriched["tracking_passer_name"])
    enriched["passer_position"] = enriched["passer_position"].fillna(enriched["tracking_passer_position"])
    enriched = enriched.drop(columns=["tracking_passer_nflId","tracking_passer_name","tracking_passer_position",])
    enriched["threw_to_best_yards_option"] = (enriched["actual_target_nflId"].eq(enriched["best_yards_option_nflId"]))
    for percentile in ("p25", "p50", "p75"):
        best_col = f"best_yards_option_{percentile}_nflId"
        if best_col in enriched.columns:
            enriched[f"threw_to_best_yards_option_{percentile}"] = enriched["actual_target_nflId"].eq(enriched[best_col])
    enriched["threw_to_best_epa_option"] = enriched["actual_target_nflId"].eq(enriched["best_epa_option_nflId"])
    enriched["positive_missed_epa"] = enriched["missed_predicted_epa"].clip(lower=0)
    enriched["positive_missed_yards"] = enriched["missed_expected_yards"].clip(lower=0)
    for percentile in ("p25", "p50", "p75"):
        missed_col=f"missed_expected_yards_{percentile}"
        timing_col=f"timing_loss_expected_yards_{percentile}"
        if missed_col in enriched.columns:
            enriched[f"positive_missed_yards_{percentile}"] = enriched[missed_col].clip(lower=0)
        if timing_col in enriched.columns:
            enriched[f"positive_timing_loss_yards_{percentile}"] = enriched[timing_col].clip(lower=0)
    enriched["high_miss_epa"] = enriched["missed_predicted_epa"].ge(1.0)
    enriched["high_miss_yards"] = enriched["missed_expected_yards"].ge(5.0)
    return enriched

def infer_passers_from_tracking(
    summary: pd.DataFrame, plays: pd.DataFrame, players: pd.DataFrame, data_dir: Path
) -> pd.DataFrame:
    needed = summary[["gameId","playId","throw_frame_id"]].drop_duplicates()
    needed = needed.rename(columns={"throw_frame_id": "frameId"})
    needed = needed.merge(plays[["gameId", "playId", "possessionTeam"]], on=["gameId", "playId"], how="left")

    tracking_parts=[]
    usecols = ["gameId","playId","frameId","nflId","club"]
    for path in sorted(data_dir.glob("tracking_week_*.csv")):
        tracking = read_csv(path, usecols=usecols)
        tracking = tracking.merge(needed, on=["gameId", "playId", "frameId"], how="inner")
        if not tracking.empty:
            tracking_parts.append(tracking)
    if not tracking_parts:
        return pd.DataFrame(
            columns=[
                "gameId",
                "playId",
                "tracking_passer_nflId",
                "tracking_passer_name",
                "tracking_passer_position",
            ]
        )

    tracking = pd.concat(tracking_parts, ignore_index=True)
    qbs = tracking.merge(players, on="nflId", how="left")
    qbs = qbs.loc[qbs["club"].eq(qbs["possessionTeam"])& qbs["position"].eq("QB")].copy()
    qbs = qbs.drop_duplicates(["gameId","playId"])
    return qbs[["gameId", "playId", "nflId", "displayName", "position"]].rename(columns={"nflId": "tracking_passer_nflId","displayName": "tracking_passer_name","position": "tracking_passer_position",})


def group_decision_summary(df: pd.DataFrame, group_cols: list[str],min_throws: int) -> pd.DataFrame:
    aggregations = {
        "throws": ("playId", "size"),
        "completion_rate": ("hadPassReception", "mean"),
        "interception_rate": ("is_interception", "mean"),
        "actual_yards_per_throw": ("receivingYards", "mean"),
        "actual_epa_per_throw": ("actual_epa", "mean"),
        "actual_target_xyards": ("actual_target_expected_yards", "mean"),
        "best_available_xyards": ("best_available_expected_yards", "mean"),
        "missed_xyards": ("missed_expected_yards", "mean"),
        "positive_missed_xyards": ("positive_missed_yards", "mean"),
        "actual_target_pred_epa": ("actual_target_predicted_epa", "mean"),
        "best_available_pred_epa": ("best_available_predicted_epa", "mean"),
        "missed_pred_epa": ("missed_predicted_epa", "mean"),
        "positive_missed_pred_epa": ("positive_missed_epa", "mean"),
        "best_yards_choice_rate": ("threw_to_best_yards_option", "mean"),
        "best_epa_choice_rate": ("threw_to_best_epa_option", "mean"),
        "acceptable_epa_choice_rate": ("acceptable_epa_choice", "mean"),
        "result_yards_over_expected": ("result_yards_over_expected", "mean"),
        "result_epa_over_predicted": ("result_epa_over_predicted", "mean"),
        "high_miss_epa_rate": ("high_miss_epa", "mean"),
        "high_miss_yards_rate": ("high_miss_yards", "mean"),
    }
    for percentile in ("p25","p50","p75"):
        optional_metrics = {
            f"actual_target_xyards_{percentile}": (f"actual_target_expected_yards_{percentile}","mean",),
            f"best_available_xyards_{percentile}": (f"best_available_expected_yards_{percentile}","mean",),
            f"missed_xyards_{percentile}": (f"missed_expected_yards_{percentile}","mean",),
            f"positive_missed_xyards_{percentile}": (f"positive_missed_yards_{percentile}","mean",),
            f"best_choice_rate_{percentile}": (f"threw_to_best_yards_option_{percentile}","mean",),
            f"acceptable_choice_rate_{percentile}": (f"acceptable_yards_choice_{percentile}","mean",),
            f"timing_loss_xyards_{percentile}": (f"timing_loss_expected_yards_{percentile}","mean",),
            f"positive_timing_loss_xyards_{percentile}": (f"positive_timing_loss_yards_{percentile}","mean",),
            f"right_receiver_right_time_rate_{percentile}": (f"right_receiver_right_time_{percentile}","mean",),
            f"acceptable_timing_rate_{percentile}": (f"acceptable_timing_choice_{percentile}","mean",),
        }
        for name, (column, func) in optional_metrics.items():
            if column in df.columns:
                aggregations[name] =(column,func)

    grouped =df.groupby(group_cols, dropna=False).agg(**aggregations)
    grouped =grouped.reset_index()
    grouped=grouped.loc[grouped["throws"].ge(min_throws)].copy()
    return grouped.sort_values(["missed_xyards","missed_pred_epa"],ascending=False)



def summarize_pressure_and_coverage(receiver_frames: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    actual_throw_frames =receiver_frames.loc[receiver_frames["is_throw_frame"].eq(True)& receiver_frames["wasTargettedReceiver"].eq(1),["gameId","playId","nearest_rusher_distance","rushers_within_3_yards","rushers_within_5_yards","throwing_lane_defender_count","throwing_lane_near_catch_count",
            "catch_point_defenders_with_arrival_chance","catch_point_nearest_defender_distance",],].drop_duplicates(["gameId", "playId"])
    df = summary.merge(actual_throw_frames, on=["gameId", "playId"],how="left")
    df["pressure_bucket"] = pd.cut(df["nearest_rusher_distance"],bins=[-np.inf, 3, 5, 8, np.inf],labels=["0-3 yards", "3-5 yards", "5-8 yards", "8+ yards"],)
    df["lane_bucket"] = pd.cut(df["throwing_lane_defender_count"],bins=[-np.inf, 0, 1, 2, np.inf],labels=["0", "1", "2", "3+"],)
    pressure = group_decision_summary(df,["pressure_bucket"], min_throws=1)
    lane = group_decision_summary(df,["lane_bucket"], min_throws=1)
    pressure["split_type"] = "nearest_rusher_distance"
    lane["split_type"] = "throwing_lane_defender_count"
    return pd.concat([pressure, lane],ignore_index=True)




def find_top_missed_plays(df: pd.DataFrame, split: str, n: int = 50) -> pd.DataFrame:
    columns = ["gameId","playId","split","passer_name","possessionTeam","defensiveTeam","down",
        "yardsToGo","timeToThrow","actual_target_name","best_yards_option_name","best_epa_option_name",
        "actual_target_expected_yards","actual_target_expected_yards_low","actual_target_expected_yards_high",
        "actual_target_interception_yards_risk","best_available_expected_yards","best_available_expected_yards_low",
        "best_available_expected_yards_high","best_available_interception_yards_risk","missed_expected_yards",
        "actual_target_predicted_epa","best_available_predicted_epa","missed_predicted_epa",
        "actual_epa","passResult","pff_passCoverage","pff_manZone"]
    columns = [column for column in columns if column in df.columns]
    return (df.loc[df["split"].eq(split), columns].sort_values(["missed_expected_yards", "missed_predicted_epa"], ascending=False).head(n))


def build_time_curve(receiver_frames: pd.DataFrame) -> pd.DataFrame:
    frames = receiver_frames.copy()
    frames["time_bucket"] = (frames["seconds_since_snap"]*2).round()/2
    best_by_bucket = frames.groupby(["split", "gameId", "playId", "time_bucket"]).agg(best_expected_yards=("expected_yards", "max"),best_predicted_epa=("predicted_epa", "max"),)
    return (best_by_bucket.reset_index().groupby(["split", "time_bucket"]).agg(plays=("playId", "nunique"),avg_best_expected_yards=("best_expected_yards", "mean"),avg_best_predicted_epa=("best_predicted_epa", "mean"),).reset_index().sort_values(["split", "time_bucket"]))






def calibration_table(df: pd.DataFrame, prob_col: str, label_col: str) -> pd.DataFrame:
    actual = df[[prob_col, label_col, "split"]].dropna().copy()
    actual["prob_bucket"] = pd.cut(actual[prob_col],bins=[i / 10 for i in range(11)],include_lowest=True,)
    out = (actual.groupby(["split", "prob_bucket"], observed=False).agg(rows=(label_col, "size"),avg_predicted=(prob_col, "mean"),actual_rate=(label_col, "mean"),).reset_index())
    out["calibration_error"] = (out["avg_predicted"] - out["actual_rate"]).abs()

    return out


def main() -> None:
    args = parse_args()
    analysis_dir = args.output_dir/"analysis"
    analysis_dir.mkdir(parents=True,exist_ok=True)
    summary=read_csv(args.output_dir/"play_decision_summary.csv")
    receiver_frames=read_csv(args.output_dir/"receiver_expected_yards.csv")
    metrics_path =args.output_dir/"model_metrics.json"
    metrics =json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

    enriched = enrich_play_summary(summary, args.data_dir)
    enriched.to_csv(analysis_dir / "enriched_play_decision_summary.csv", index=False)

    all_summary= group_decision_summary(enriched, ["split"], min_throws=1)
    team_summary = group_decision_summary(enriched, ["split","possessionTeam"],args.min_throws)
    qb_summary = group_decision_summary(enriched, ["split","passer_name"],args.min_throws)
    qb_summary_all= group_decision_summary(enriched, ["passer_name"],args.min_throws)
    coverage_summary = group_decision_summary(enriched, ["split","pff_manZone"],args.min_throws)
    pressure_lane_summary =summarize_pressure_and_coverage(receiver_frames, enriched)
    time_curve =build_time_curve(receiver_frames)
    actual_throws = receiver_frames.loc[receiver_frames["is_throw_frame"].eq(True)& receiver_frames["wasTargettedReceiver"].eq(1)].copy()
    completion_calibration = calibration_table(actual_throws, "completion_probability", "hadPassReception")
    interception_calibration = calibration_table(actual_throws, "interception_probability", "is_interception")
    all_summary.to_csv(analysis_dir/"overall_summary.csv", index=False)
    team_summary.to_csv(analysis_dir/"team_summary.csv", index=False)
    qb_summary.to_csv(analysis_dir/"qb_summary.csv", index=False)
    qb_summary_all.to_csv(analysis_dir/"qb_summary_all.csv", index=False)
    coverage_summary.to_csv(analysis_dir/"coverage_summary.csv", index=False)
    pressure_lane_summary.to_csv(analysis_dir/"pressure_lane_summary.csv", index=False)
    time_curve.to_csv(analysis_dir/"time_curve_summary.csv", index=False)
    completion_calibration.to_csv(analysis_dir/"completion_calibration.csv", index=False)
    interception_calibration.to_csv(analysis_dir/"interception_calibration.csv", index=False)
    find_top_missed_plays(enriched, "test").to_csv(analysis_dir / "top_missed_plays_test.csv",index=False)
    find_top_missed_plays(enriched, "train").to_csv(analysis_dir / "top_missed_plays_train.csv",index=False)

    print("Model metrics")
    for key in ["train_rows","test_rows","completion_auc","interception_auc","epa_mae","yards_mae",]:print(f"  {key}: {metrics.get(key)}")
    print("\nOverall decision summary")
    print(all_summary.to_string(index=False))
    print(f"\nWrote analysis tables to {analysis_dir}")


if __name__ == "__main__":
    main()
