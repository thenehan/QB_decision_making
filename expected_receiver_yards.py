"""Expected receiver yards."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

#global
FIELD_WIDTH = 53.3
FIELD_LENGTH = 120.0
OFFENSIVE_GOAL_X = 110.0
FRAMES_PER_SECOND = 10.0

TREE_MODEL_FAMILY = "xgboost"
COMPLETION_CALIBRATION_WEIGHT = 0.30
INTERCEPTION_CALIBRATION_WEIGHT = 0.40
COMPLETION_SMOOTHING = 20.0
INTERCEPTION_SMOOTHING = 100.0
SIMILARITY_FEATURE_COLUMNS = [
    "seconds_since_snap","target_depth","target_distance_from_qb","target_width_from_middle",
    "target_depth_past_sticks","nearest_rusher_distance","throwing_lane_defender_count",
    "catch_point_nearest_defender_distance","catch_point_defenders_within_5","catch_point_blocker_advantage_5",
    "defender_1_distance",
]

PASS_EVENTS = {"pass_forward", "pass_shovel"}
DEAD_EVENTS = {"qb_sack","qb_strip_sack","qb_slide","tackle","out_of_bounds","touchdown","fumble","fumble_offense_recovered","fumble_defense_recovered",}
FEATURE_COLUMNS = ["seconds_since_snap","down","yardsToGo","pre_snap_ep","receiver_x","receiver_y","receiver_speed",
    "receiver_accel","receiver_dir_sin","receiver_dir_cos","receiver_vx","receiver_vy","yards_past_los","yards_to_goal",
    "sideline_distance","qb_x","qb_y","qb_speed","qb_orientation_sin","qb_orientation_cos","qb_facing_receiver_cos",
    "qb_facing_target_cos","target_front_side","throw_distance","estimated_ball_travel_time","target_point_x",
    "target_point_y","target_point_distance_from_receiver","target_depth","target_width_from_middle","target_y_from_middle",
    "target_yards_to_goal","target_distance_from_qb","target_depth_past_sticks","target_behind_los","target_short",
    "target_intermediate","target_deep","projected_receiver_x","projected_receiver_y","projected_yards_past_los",
    "nearest_rusher_distance","nearest_rusher_closing_speed","time_to_pressure","pressure_score","is_under_pressure",
    "rushers_within_3_yards","rushers_within_5_yards","defender_1_distance","defender_1_x_diff","defender_1_y_diff",
    "defender_1_speed","defender_1_closing_speed","defender_2_distance","defender_2_x_diff","defender_2_y_diff",
    "defender_2_speed","defender_2_closing_speed","defender_3_distance","defender_3_x_diff","defender_3_y_diff",
    "defender_3_speed","defender_3_closing_speed","defender_4_distance","defender_4_x_diff","defender_4_y_diff",
    "defender_4_speed","defender_4_closing_speed","throwing_lane_defender_count","throwing_lane_closest_distance",
    "throwing_lane_downfield_defender_count","throwing_lane_width","throwing_lane_near_qb_count","throwing_lane_near_catch_count",
    "throwing_lane_facing_qb_count","throwing_lane_midfield_defender_count","throwing_lane_closest_progress",
    "throwing_lane_closest_qb_distance","throwing_lane_density","receiver_lane_separation_gap",
    "catch_point_nearest_defender_distance","catch_point_nearest_defender_time",
    "catch_point_defender_arrival_margin","catch_point_defenders_with_arrival_chance","catch_point_defenders_within_3",
    "catch_point_defenders_within_5","catch_point_defenders_within_8","catch_point_nearest_defender_x_diff","catch_point_nearest_defender_y_diff",
    "catch_point_open_grass_yards","catch_point_blockers_within_5","catch_point_blockers_within_8","catch_point_blocker_advantage_5",]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate expected yards for each receiver at each pre-throw frame.")
    parser.add_argument("--data-dir", type=Path,default=Path("."))
    parser.add_argument("--output-dir", type=Path,default=Path("expected_yards_output"))
    parser.add_argument("--weeks",nargs="+",type=int,default=list(range(1, 10)),help="Tracking weeks to process, for example: --weeks 1 2 3",)
    parser.add_argument("--min-play-seconds",type=float,default=2.0,help="Keep plays whose snap-to-pass/dead interval is longer than this.",)
    parser.add_argument("--max-plays",type=int,default=None,help="Optional limit for quick local experiments.",)
    parser.add_argument("--random-state",type=int,default=42,help="Random seed used for train/test split and models.",)
    parser.add_argument("--test-size",type=float,default=0.2,help="Fraction of eligible plays held out for test evaluation.",)
    parser.add_argument("--test-weeks", nargs="+", type=int, default=None)
    parser.add_argument("--acceptable-yards-gap", type=float, default=1.0)
    parser.add_argument("--acceptable-yards-pct", type=float, default=0.20)
    parser.add_argument("--acceptable-epa-gap", type=float, default=0.25)
    parser.add_argument("--interception-yard-penalty",type=float,default=-45.0,help="Yard value assigned to an interception in risk-adjusted expected yards.",)
    parser.add_argument("--similar-throws",type=int,default=100,help="Number of nearest historical throws used for Bayesian completion uncertainty.",)
    parser.add_argument("--bayes-model-strength",type=float,default=25.0,help="Pseudo-sample strength assigned to the tree-model completion prediction.",)
    parser.add_argument("--interception-positive-weight-cap", type=float, default=50.0)
    return parser.parse_args()


def read_csv(data_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path =data_dir/name
    if not path.exists():
        raise FileNotFoundError(f"Could not find{path}")
    return pd.read_csv(path,**kwargs)


def normalize_tracking_coordinates(df: pd.DataFrame)-> pd.DataFrame:
    """Flip play direction."""
    out = df.copy()
    going_right = out["playDirection"].eq("right")
    out["x_norm"] = np.where(going_right, out["x"], FIELD_LENGTH - out["x"])
    out["y_norm"] = np.where(going_right, out["y"], FIELD_WIDTH - out["y"])


    direction_radians = np.deg2rad(out["dir"].fillna(0.0))
    vx = out["s"].fillna(0.0)*np.sin(direction_radians)
    vy = out["s"].fillna(0.0)*np.cos(direction_radians)
    out["vx_norm"] = np.where(going_right,vx,-vx)
    out["vy_norm"] = np.where(going_right,vy,-vy)

    out["dir_norm_radians"]=np.arctan2(out["vx_norm"], out["vy_norm"])
    out["dir_sin"]=np.sin(out["dir_norm_radians"])
    out["dir_cos"]=np.cos(out["dir_norm_radians"])
    orientation_radians = np.deg2rad(out["o"].fillna(out["dir"]).fillna(0.0))
    ox = np.sin(orientation_radians)
    oy = np.cos(orientation_radians)
    out["orientation_x_norm"] = np.where(going_right, ox, -ox)
    out["orientation_y_norm"] = np.where(going_right, oy, -oy)
    out["orientation_norm_radians"] = np.arctan2(out["orientation_x_norm"], out["orientation_y_norm"])
    out["orientation_sin"] = np.sin(out["orientation_norm_radians"])
    out["orientation_cos"] = np.cos(out["orientation_norm_radians"])
    return out




def normalized_line_of_scrimmage(play: pd.Series, play_direction: str) -> float:
    los = float(play["absoluteYardlineNumber"])
    if play_direction == "left":
        los = FIELD_LENGTH-los
    return los

def first_frame_with_event(play_tracking:pd.DataFrame,events: set[str])-> int | None:
    event_frames = play_tracking.loc[play_tracking["event"].isin(events), "frameId"]
    if event_frames.empty:
        return None
    return int(event_frames.min())


def choose_end_frame(play_tracking:pd.DataFrame)-> tuple[int|None,str]:
    pass_frame = first_frame_with_event(play_tracking, PASS_EVENTS)
    if pass_frame is not None:
        return pass_frame, "pass"
    dead_frame = first_frame_with_event(play_tracking, DEAD_EVENTS)
    if dead_frame is not None:
        return dead_frame, "dead"
    return None,"missing"


def defender_features(receivers: pd.DataFrame, defenders: pd.DataFrame) -> pd.DataFrame:
    if receivers.empty:
        return receivers
    receiver_cols=["gameId","playId","frameId","nflId","x_norm","y_norm","vx_norm","vy_norm"]
    defender_cols=["frameId","nflId","x_norm","y_norm","s","vx_norm","vy_norm"]

    pairs = receivers[receiver_cols].merge(defenders[defender_cols],on="frameId",suffixes=("_receiver", "_defender"),)
    if pairs.empty:
        receivers = receivers.copy()
        for rank in range(1, 5):
            receivers[f"defender_{rank}_distance"]= np.nan
            receivers[f"defender_{rank}_x_diff"]= np.nan
            receivers[f"defender_{rank}_y_diff"]= np.nan
            receivers[f"defender_{rank}_speed"]= np.nan
            receivers[f"defender_{rank}_closing_speed"]= np.nan
        receivers["throwing_lane_defender_count"]= 0
        receivers["throwing_lane_closest_distance"]= np.nan
        receivers["throwing_lane_downfield_defender_count"]= 0
        return receivers

    pairs["defender_x_diff"] =pairs["x_norm_defender"]-pairs["x_norm_receiver"]
    pairs["defender_y_diff"] =pairs["y_norm_defender"]-pairs["y_norm_receiver"]
    pairs["defender_distance"] = np.hypot(pairs["defender_x_diff"],pairs["defender_y_diff"])

    dx = pairs["x_norm_receiver"]-pairs["x_norm_defender"]
    dy = pairs["y_norm_receiver"]-pairs["y_norm_defender"]
    distance = pairs["defender_distance"].replace(0, np.nan)
    relative_vx = pairs["vx_norm_defender"]-pairs["vx_norm_receiver"]
    relative_vy = pairs["vy_norm_defender"]-pairs["vy_norm_receiver"]
    pairs["defender_closing_speed"]= (relative_vx*dx+ relative_vy*dy)/ distance

    group_cols =["gameId","playId","frameId","nflId_receiver"]
    pairs["defender_rank"] = pairs.groupby(group_cols)["defender_distance"].rank(method="first")
    nearest_four = pairs.loc[pairs["defender_rank"].le(4)].copy()
    nearest_four["defender_rank"] = nearest_four["defender_rank"].astype(int)
    nearest_four = nearest_four[["gameId","playId","frameId","nflId_receiver","defender_rank","defender_distance",
            "defender_x_diff","defender_y_diff","s","defender_closing_speed"]]
    wide_parts = []


    for source_col, output_name in [("defender_distance", "distance"),("defender_x_diff", "x_diff"),("defender_y_diff", "y_diff"),
        ("s", "speed"),("defender_closing_speed", "closing_speed")]:
        wide = nearest_four.pivot(index=group_cols, columns="defender_rank", values=source_col)
        wide.columns=[f"defender_{rank}_{output_name}" for rank in wide.columns]
        wide_parts.append(wide)
    nearest = pd.concat(wide_parts, axis=1).reset_index().rename(columns={"nflId_receiver": "nflId"})
    return receivers.merge(nearest,on=["gameId","playId","frameId","nflId"],how="left")



def add_throwing_lane_features(receivers: pd.DataFrame, defenders: pd.DataFrame) -> pd.DataFrame:
    """Count lane defenders."""
    if receivers.empty:
        return receivers

    out = receivers.copy()
    if defenders.empty:
        out["throwing_lane_defender_count"]=0
        out["throwing_lane_closest_distance"]= 6.0
        out["throwing_lane_downfield_defender_count"]=0
        out["throwing_lane_width"]= np.clip(1.5 + out["throw_distance"]/ 25.0,2.0, 5.0)
        out["throwing_lane_near_qb_count"] =0
        out["throwing_lane_near_catch_count"] =0
        out["throwing_lane_facing_qb_count"]= 0
        out["throwing_lane_midfield_defender_count"] = 0
        out["throwing_lane_closest_progress"] = 1.0
        out["throwing_lane_closest_qb_distance"] = out["target_distance_from_qb"]
        out["throwing_lane_density"] = 0.0
        return out

    receiver_cols = ["gameId","playId","frameId","nflId","qb_x","qb_y","target_point_x","target_point_y",]
    defender_cols = ["frameId","x_norm","y_norm","vx_norm","vy_norm"]
    lane =out[receiver_cols].merge(defenders[defender_cols],on="frameId")

    lane_dx = lane["target_point_x"]-lane["qb_x"]
    lane_dy = lane["target_point_y"]-lane["qb_y"]
    lane_len_sq = lane_dx.pow(2)+lane_dy.pow(2)
    lane_length = np.sqrt(lane_len_sq)
    defender_dx = lane["x_norm"]-lane["qb_x"]
    defender_dy = lane["y_norm"]-lane["qb_y"]
    lane["throwing_lane_width"]= np.clip(1.5+lane_length/25.0,2.0,5.0)
    lane["lane_progress"] = ((defender_dx * lane_dx + defender_dy * lane_dy) / lane_len_sq.replace(0, np.nan))
    projection_x= lane["qb_x"]+lane["lane_progress"]* lane_dx
    projection_y= lane["qb_y"]+lane["lane_progress"]* lane_dy
    lane["lane_distance"] = np.hypot(lane["x_norm"]-projection_x,lane["y_norm"]-projection_y)
    qb_distance = np.hypot(lane["x_norm"]-lane["qb_x"],lane["y_norm"]-lane["qb_y"])
    catch_distance = np.hypot(lane["x_norm"] - lane["target_point_x"],lane["y_norm"] - lane["target_point_y"],)
    toward_qb_x=lane["qb_x"]-lane["x_norm"]
    toward_qb_y=lane["qb_y"]-lane["y_norm"]
    toward_qb_len = np.hypot(toward_qb_x,toward_qb_y).replace(0,np.nan)
    defender_speed = np.hypot(lane["vx_norm"],lane["vy_norm"]).replace(0, np.nan)
    lane["facing_qb"] = ((lane["vx_norm"] * toward_qb_x + lane["vy_norm"] * toward_qb_y)/(defender_speed * toward_qb_len)).gt(0.25)

    in_lane = lane.loc[lane["lane_progress"].between(0.0, 1.0)&lane["lane_distance"].le(lane["throwing_lane_width"])].copy()
    if in_lane.empty:
        out["throwing_lane_defender_count"] = 0
        out["throwing_lane_closest_distance"] = 6.0
        out["throwing_lane_downfield_defender_count"] = 0
        out["throwing_lane_width"] = np.clip(1.5 + out["throw_distance"] / 25.0, 2.0, 5.0)
        out["throwing_lane_near_qb_count"] = 0
        out["throwing_lane_near_catch_count"] = 0
        out["throwing_lane_facing_qb_count"] = 0
        out["throwing_lane_midfield_defender_count"] = 0
        out["throwing_lane_closest_progress"] = 1.0
        out["throwing_lane_closest_qb_distance"] = out["target_distance_from_qb"]
        out["throwing_lane_density"] = 0.0
        return out

    in_lane["qb_distance"]=qb_distance.loc[in_lane.index]
    in_lane["catch_distance"]=catch_distance.loc[in_lane.index]
    group_cols = ["gameId", "playId","frameId", "nflId"]
    closest_idx = in_lane.groupby(group_cols)["lane_distance"].idxmin()
    closest_lane = in_lane.loc[closest_idx, group_cols + ["lane_progress", "qb_distance"]].rename(columns={"lane_progress": "throwing_lane_closest_progress", "qb_distance": "throwing_lane_closest_qb_distance"})
    lane_summary = in_lane.groupby(group_cols).agg(throwing_lane_defender_count=("lane_distance","size"),throwing_lane_closest_distance=("lane_distance", "min"),throwing_lane_downfield_defender_count=("lane_progress",lambda progress: int((progress >= 0.35).sum())),throwing_lane_width=("throwing_lane_width", "first"),throwing_lane_near_qb_count=("qb_distance", lambda distance: int((distance <= 5.0).sum())),throwing_lane_near_catch_count=("catch_distance",lambda distance: int((distance <= 5.0).sum())),throwing_lane_facing_qb_count=("facing_qb", "sum"),throwing_lane_midfield_defender_count=("lane_progress", lambda progress: int(((progress > 0.15) & (progress < 0.85)).sum())))
    lane_summary = lane_summary.reset_index().merge(closest_lane, on=group_cols, how="left")
    out = out.merge(lane_summary, on=group_cols, how="left")
    out["throwing_lane_defender_count"] =out["throwing_lane_defender_count"].fillna(0)
    out["throwing_lane_closest_distance"]= out["throwing_lane_closest_distance"].fillna(
        6.0
    )
    out["throwing_lane_downfield_defender_count"]= (out["throwing_lane_downfield_defender_count"].fillna(0))
    out["throwing_lane_width"]= out["throwing_lane_width"].fillna(np.clip(1.5 + out["throw_distance"] / 25.0, 2.0, 5.0))
    out["throwing_lane_near_qb_count"]= out["throwing_lane_near_qb_count"].fillna(0)
    out["throwing_lane_near_catch_count"]= out["throwing_lane_near_catch_count"].fillna(0)
    out["throwing_lane_facing_qb_count"]= out["throwing_lane_facing_qb_count"].fillna(0)
    out["throwing_lane_midfield_defender_count"] = out["throwing_lane_midfield_defender_count"].fillna(0)
    out["throwing_lane_closest_progress"] = out["throwing_lane_closest_progress"].fillna(1.0)
    out["throwing_lane_closest_qb_distance"] = out["throwing_lane_closest_qb_distance"].fillna(out["target_distance_from_qb"])
    out["throwing_lane_density"] = out["throwing_lane_defender_count"] / np.maximum(out["target_distance_from_qb"], 1.0)

    return out


def add_pressure_features(receivers: pd.DataFrame, defenders: pd.DataFrame) -> pd.DataFrame:
    if receivers.empty:
        return receivers
    out = receivers.copy()
    if defenders.empty:
        out["nearest_rusher_distance"]= np.nan
        out["nearest_rusher_closing_speed"]= np.nan
        out["time_to_pressure"] =5.0
        out["pressure_score"]=0.0
        out["is_under_pressure"]=0
        out["rushers_within_3_yards"]=0
        out["rushers_within_5_yards"]=0

        return out

    pressure = out[
        ["gameId", "playId", "frameId", "nflId", "qb_x", "qb_y"]].merge(defenders[["frameId", "x_norm", "y_norm", "vx_norm", "vy_norm"]],on="frameId")
    pressure["rusher_distance"] = np.hypot(pressure["x_norm"] - pressure["qb_x"], pressure["y_norm"] - pressure["qb_y"])
    to_qb_x = pressure["qb_x"]-pressure["x_norm"]
    to_qb_y = pressure["qb_y"]-pressure["y_norm"]
    distance = pressure["rusher_distance"].replace(0, np.nan)
    pressure["rusher_closing_speed"] =(pressure["vx_norm"] * to_qb_x + pressure["vy_norm"] * to_qb_y)/ distance

    group_cols = ["gameId", "playId","frameId","nflId"]
    summary = pressure.groupby(group_cols).agg(nearest_rusher_distance=("rusher_distance", "min"),rushers_within_3_yards=("rusher_distance", lambda distance: int((distance <= 3.0).sum())),rushers_within_5_yards=("rusher_distance", lambda distance: int((distance <= 5.0).sum())))
    nearest = pressure.loc[pressure.groupby(group_cols)["rusher_distance"].idxmin()]
    closing = nearest[group_cols + ["rusher_closing_speed"]].rename(columns={"rusher_closing_speed": "nearest_rusher_closing_speed"})
    summary = summary.reset_index().merge(closing,on=group_cols, how="left")
    summary["nearest_rusher_closing_speed"] = summary["nearest_rusher_closing_speed"].fillna(0.0)
    closing_speed = summary["nearest_rusher_closing_speed"].clip(lower=0.1)
    summary["time_to_pressure"] = ((summary["nearest_rusher_distance"] - 1.5).clip(lower=0.0) / closing_speed).clip(0.0, 5.0)
    summary["pressure_score"] = (1.0 / (summary["nearest_rusher_distance"].clip(lower=0.5))+ 0.25 * summary["rushers_within_3_yards"]+ 0.15 * summary["rushers_within_5_yards"]+ 0.10 * summary["nearest_rusher_closing_speed"].clip(lower=0.0))
    summary["is_under_pressure"] = (summary["nearest_rusher_distance"].le(3.0) | summary["time_to_pressure"].le(1.0) | summary["rushers_within_3_yards"].ge(1)).astype(int)
    
    return out.merge(summary, on=group_cols, how="left")


def add_catch_point_features(
    receivers: pd.DataFrame, defenders: pd.DataFrame, offense: pd.DataFrame
) -> pd.DataFrame:
    if receivers.empty:
        return receivers

    out = receivers.copy()
    if defenders.empty:
        out["catch_point_nearest_defender_distance"]=np.nan
        out["catch_point_nearest_defender_time"]=np.nan
        out["catch_point_defender_arrival_margin"]=np.nan
        out["catch_point_defenders_with_arrival_chance"] = 0
        out["catch_point_defenders_within_3"] = 0
        out["catch_point_defenders_within_5"] = 0
        out["catch_point_defenders_within_8"] = 0
        out["catch_point_nearest_defender_x_diff"]= np.nan
        out["catch_point_nearest_defender_y_diff"]= np.nan
        out["catch_point_open_grass_yards"] = 0
        out["catch_point_blockers_within_5"] =0
        out["catch_point_blockers_within_8"] = 0
        out["catch_point_blocker_advantage_5"] = 0
        return out

    catch = out[["gameId","playId","frameId","nflId","target_point_x","target_point_y","estimated_ball_travel_time","target_yards_to_goal",]].merge(defenders[["frameId", "x_norm", "y_norm", "s"]],on="frameId")
    catch["catch_point_defender_distance"] = np.hypot(catch["x_norm"] - catch["target_point_x"],catch["y_norm"] - catch["target_point_y"])
    effective_speed = catch["s"].clip(lower=1.0)
    catch["catch_point_defender_time"] = catch["catch_point_defender_distance"]/effective_speed
    catch["catch_point_arrival_margin"] = (catch["catch_point_defender_time"] - catch["estimated_ball_travel_time"])

    group_cols = ["gameId", "playId", "frameId", "nflId"]
    catch["catch_point_defender_x_diff"]= catch["x_norm"]-catch["target_point_x"]
    catch["catch_point_defender_y_diff"]= catch["y_norm"]-catch["target_point_y"]
    nearest = catch.loc[catch.groupby(group_cols)["catch_point_defender_distance"].idxmin()]
    summary = nearest[group_cols + ["catch_point_defender_distance","catch_point_defender_time","catch_point_arrival_margin","catch_point_defender_x_diff","catch_point_defender_y_diff","target_yards_to_goal"]].rename(columns={"catch_point_defender_distance": "catch_point_nearest_defender_distance","catch_point_defender_time": "catch_point_nearest_defender_time","catch_point_arrival_margin": "catch_point_defender_arrival_margin","catch_point_defender_x_diff": "catch_point_nearest_defender_x_diff","catch_point_defender_y_diff": "catch_point_nearest_defender_y_diff","target_yards_to_goal": "catch_point_open_grass_yards"})
    arrival_counts = catch.groupby(group_cols).agg(catch_point_defenders_with_arrival_chance=("catch_point_arrival_margin",
            lambda margin: int((margin <= 0.30).sum())),
        catch_point_defenders_within_3=("catch_point_defender_distance",
            lambda distance: int((distance <= 3.0).sum())),
        catch_point_defenders_within_5=("catch_point_defender_distance",
            lambda distance: int((distance <= 5.0).sum())),
        catch_point_defenders_within_8=("catch_point_defender_distance",
            lambda distance: int((distance <= 8.0).sum())),
    )
    summary = summary.merge(arrival_counts.reset_index(),on=group_cols, how="left")
    out = out.merge(summary,on=group_cols, how="left")

    blockers = offense[["gameId", "playId", "frameId", "nflId", "x_norm", "y_norm"]].copy()
    blocker_pairs = out[["gameId", "playId", "frameId", "nflId", "target_point_x", "target_point_y"]].merge(blockers,on=["gameId", "playId", "frameId"],suffixes=("_receiver", "_blocker"))
    blocker_pairs = blocker_pairs.loc[blocker_pairs["nflId_receiver"].ne(blocker_pairs["nflId_blocker"])].copy()
    if blocker_pairs.empty:
        out["catch_point_blockers_within_5"] = 0
        out["catch_point_blockers_within_8"] = 0
    else:
        blocker_pairs["blocker_distance"] = np.hypot(
            blocker_pairs["x_norm"]-blocker_pairs["target_point_x"],
            blocker_pairs["y_norm"]-blocker_pairs["target_point_y"])
        
        blocker_summary = blocker_pairs.groupby(["gameId", "playId", "frameId", "nflId_receiver"]).agg(catch_point_blockers_within_5=("blocker_distance",
                lambda distance: int((distance <= 5.0).sum()),
            ),
            catch_point_blockers_within_8=("blocker_distance",
                lambda distance: int((distance <= 8.0).sum()),
            ))
        blocker_summary = blocker_summary.reset_index().rename(columns={"nflId_receiver": "nflId"})
        out = out.merge(blocker_summary, on=group_cols, how="left")
    out["catch_point_blockers_within_5"] =out["catch_point_blockers_within_5"].fillna(0)
    out["catch_point_blockers_within_8"] =out["catch_point_blockers_within_8"].fillna(0)
    out["catch_point_blocker_advantage_5"] = (out["catch_point_blockers_within_5"]-out["catch_point_defenders_within_5"])
    
    return out


def add_qb_features(receivers: pd.DataFrame, offense: pd.DataFrame) -> pd.DataFrame:
    qbs = offense.loc[offense["position"].eq("QB"),["gameId", "playId", "frameId", "x_norm", "y_norm", "s", "orientation_sin", "orientation_cos"]].rename(columns={"x_norm": "qb_x", "y_norm": "qb_y", "s": "qb_speed", "orientation_sin": "qb_orientation_sin", "orientation_cos": "qb_orientation_cos"})

    if qbs.empty:
        out = receivers.copy()
        out["qb_x"]=np.nan
        out["qb_y"]=np.nan
        out["qb_speed"]=np.nan
        out["qb_orientation_sin"]=np.nan
        out["qb_orientation_cos"]=np.nan
        return out

    qbs = qbs.drop_duplicates(["gameId","playId","frameId"])
    return receivers.merge(qbs, on=["gameId","playId","frameId"], how="left")


def build_play_features(play_tracking: pd.DataFrame,play: pd.Series,route_runners: pd.DataFrame,min_play_seconds: float) -> pd.DataFrame | None:
    snap_frame= first_frame_with_event(play_tracking, {"ball_snap"})
    end_frame, end_reason= choose_end_frame(play_tracking)
    if snap_frame is None or end_frame is None or end_frame <= snap_frame:
        return None
    play_seconds = (end_frame - snap_frame) / FRAMES_PER_SECOND
    if play_seconds <= min_play_seconds:
        return None
    frame_window = play_tracking.loc[play_tracking["frameId"].between(snap_frame, end_frame)].copy()
    if frame_window.empty:
        return None

    play_direction = frame_window["playDirection"].dropna().iloc[0]
    line_x=normalized_line_of_scrimmage(play, play_direction)
    offense_team = play["possessionTeam"]
    defense_team = play["defensiveTeam"]
    offense=frame_window.loc[frame_window["club"].eq(offense_team)].copy()
    defenders = frame_window.loc[frame_window["club"].eq(defense_team)].copy()

    receivers = offense.merge(route_runners[["gameId", "playId", "nflId", "routeRan", "wasTargettedReceiver"]],on=["gameId", "playId", "nflId"],how="inner")
    if receivers.empty:
        return None

    receivers=defender_features(receivers, defenders)
    receivers= add_qb_features(receivers, offense)
    receivers= add_pressure_features(receivers, defenders)

    receivers["seconds_since_snap"] =(receivers["frameId"]-snap_frame)/FRAMES_PER_SECOND
    receivers["line_of_scrimmage_x"]=line_x
    receivers["yards_past_los"] = receivers["x_norm"]-line_x
    receivers["yards_to_goal"] = OFFENSIVE_GOAL_X-receivers["x_norm"]
    receivers["sideline_distance"] = np.minimum(
        receivers["y_norm"], FIELD_WIDTH-receivers["y_norm"]
    )

    receivers["throw_distance"] = np.hypot(receivers["x_norm"] - receivers["qb_x"], receivers["y_norm"] - receivers["qb_y"])
    receivers["estimated_ball_travel_time"] = np.clip(0.35 + receivers["throw_distance"] / 35.0, 0.45, 1.80)
    receivers["projected_receiver_x"] = (receivers["x_norm"] + receivers["vx_norm"] * receivers["estimated_ball_travel_time"])
    receivers["projected_receiver_y"] = (receivers["y_norm"] + receivers["vy_norm"] * receivers["estimated_ball_travel_time"]).clip(0.0, FIELD_WIDTH)
    receivers["projected_yards_past_los"] = receivers["projected_receiver_x"] - line_x
    target_x = play.get("targetX",np.nan)
    target_y = play.get("targetY",np.nan)
    if pd.notna(target_x) and pd.notna(target_y):
        if play_direction == "left":
            target_x = FIELD_LENGTH-float(target_x)
            target_y = FIELD_WIDTH-float(target_y)
        else:
            target_x= float(target_x)
            target_y= float(target_y)
    use_actual_target= (receivers["frameId"].eq(end_frame)&receivers["wasTargettedReceiver"].eq(1)&pd.notna(target_x)&pd.notna(target_y)&(end_reason == "pass"))
    receivers["target_point_x"] = np.where(use_actual_target, target_x, receivers["projected_receiver_x"])
    receivers["target_point_y"] = np.where(use_actual_target, target_y, receivers["projected_receiver_y"])
    receivers["target_point_distance_from_receiver"] = np.hypot(receivers["target_point_x"] - receivers["x_norm"],receivers["target_point_y"] - receivers["y_norm"])
    qb_to_receiver_x = receivers["x_norm"] - receivers["qb_x"]
    qb_to_receiver_y = receivers["y_norm"] - receivers["qb_y"]
    qb_to_receiver_len = np.hypot(qb_to_receiver_x, qb_to_receiver_y).replace(0, np.nan)
    qb_to_target_x = receivers["target_point_x"] - receivers["qb_x"]
    qb_to_target_y = receivers["target_point_y"] - receivers["qb_y"]
    qb_to_target_len = np.hypot(qb_to_target_x, qb_to_target_y).replace(0, np.nan)
    receivers["qb_facing_receiver_cos"] = ((receivers["qb_orientation_sin"] * qb_to_receiver_x + receivers["qb_orientation_cos"] * qb_to_receiver_y) / qb_to_receiver_len)
    receivers["qb_facing_target_cos"] = ((receivers["qb_orientation_sin"] * qb_to_target_x + receivers["qb_orientation_cos"] * qb_to_target_y) / qb_to_target_len)
    receivers["target_front_side"] = receivers["qb_facing_target_cos"].ge(0.25).astype(int)
    receivers["target_depth"] = receivers["target_point_x"] - line_x
    receivers["target_width_from_middle"] = (receivers["target_point_y"] - FIELD_WIDTH / 2).abs()
    receivers["target_y_from_middle"] = receivers["target_point_y"] - FIELD_WIDTH/2
    receivers["target_yards_to_goal"] = OFFENSIVE_GOAL_X - receivers["target_point_x"]
    receivers["target_distance_from_qb"] = np.hypot(receivers["target_point_x"] - receivers["qb_x"],receivers["target_point_y"] - receivers["qb_y"])
    receivers["target_depth_past_sticks"] = receivers["target_depth"] - play["yardsToGo"]
    receivers["target_behind_los"] = receivers["target_depth"].lt(0).astype(int)
    receivers["target_short"] = receivers["target_depth"].between(0, 10, inclusive="left").astype(int)
    receivers["target_intermediate"] = receivers["target_depth"].between(10, 20, inclusive="left").astype(int)
    receivers["target_deep"] = receivers["target_depth"].ge(20).astype(int)
    receivers =add_throwing_lane_features(receivers,defenders)
    receivers["receiver_lane_separation_gap"] = receivers["defender_1_distance"].fillna(8.0) - receivers["throwing_lane_closest_distance"]
    receivers = add_catch_point_features(receivers,defenders,offense)

    receivers["down"] = play["down"]
    receivers["yardsToGo"] = play["yardsToGo"]
    receivers["pre_snap_ep"] = play["expectedPoints"]
    receivers["actual_epa"] = play["expectedPointsAdded"]
    receivers["passResult"] = play["passResult"]
    receivers["end_frame"] = end_frame
    receivers["end_reason"] = end_reason
    receivers["play_duration_seconds"] = play_seconds

    return receivers.rename(columns={"x_norm": "receiver_x","y_norm": "receiver_y","s": "receiver_speed","a": "receiver_accel","dir_sin": "receiver_dir_sin","dir_cos": "receiver_dir_cos","vx_norm": "receiver_vx","vy_norm": "receiver_vy",})


def build_feature_table(tracking: pd.DataFrame, plays: pd.DataFrame, player_play: pd.DataFrame, players: pd.DataFrame, min_play_seconds: float, max_plays: int | None,) -> pd.DataFrame:
    dropback_plays = plays.loc[plays["isDropback"].eq(True)].copy()
    route_runners = player_play.loc[player_play["wasRunningRoute"].eq(1)].copy()
    target_labels = player_play[["gameId","playId","nflId","hadPassReception","receivingYards","yardageGainedAfterTheCatch"]].copy()
    tracking = tracking.merge(players[["nflId","position"]],on="nflId", how="left")
    tracking = normalize_tracking_coordinates(tracking)
    tracking = tracking.merge(
        dropback_plays[["gameId","playId","possessionTeam","defensiveTeam","absoluteYardlineNumber","down","yardsToGo","expectedPoints","expectedPointsAdded","passResult",
                "targetX","targetY","passLength"]],on=["gameId", "playId"],how="inner",)
    if max_plays is not None:
        keep =tracking[["gameId","playId"]].drop_duplicates().head(max_plays)
        tracking=tracking.merge(keep, on=["gameId", "playId"], how="inner")
    play_lookup = dropback_plays.set_index(["gameId", "playId"])
    feature_parts=[]
    for (game_id, play_id), play_tracking in tracking.groupby(["gameId", "playId"], sort=False):
        play = play_lookup.loc[(game_id, play_id)]
        play_routes = route_runners.loc[route_runners["gameId"].eq(game_id) & route_runners["playId"].eq(play_id)]
        if play_routes.empty:
            continue

        play_features = build_play_features(play_tracking=play_tracking,play=play,route_runners=play_routes, min_play_seconds=min_play_seconds,)
        if play_features is not None:
            feature_parts.append(play_features)

    if not feature_parts:
        return pd.DataFrame()

    features = pd.concat(feature_parts, ignore_index=True)
    features = features.merge(target_labels, on=["gameId", "playId", "nflId"], how="left")
    features["wasTargettedReceiver"] = features["wasTargettedReceiver"].fillna(0).astype(int)
    features["hadPassReception"] = features["hadPassReception"].fillna(0).astype(int)
    features["receivingYards"] = features["receivingYards"].fillna(0).astype(float)
    features["yardageGainedAfterTheCatch"] = (features["yardageGainedAfterTheCatch"].fillna(0).astype(float))
    features["actual_yac"] = features["yardageGainedAfterTheCatch"].clip(lower=0)
    features["actual_air_yards"] = features["receivingYards"] - features["actual_yac"]
    features["is_throw_frame"] = features["frameId"].eq(features["end_frame"]) & features["end_reason"].eq("pass")
    features["is_interception"] = features["passResult"].eq("IN").astype(int)
    return features


def load_tracking_weeks(data_dir: Path, weeks: Iterable[int]) -> pd.DataFrame:
    week_frames = []
    usecols = ["gameId", "playId", "nflId", "displayName", "frameId", "time", "club", "playDirection", "x",
        "y", "s", "a", "o", "dir","event"]
    for week in weeks:
        week_df = read_csv(data_dir, f"tracking_week_{week}.csv",usecols=usecols)
        week_df["week"]=int(week)
        week_frames.append(week_df)
    return pd.concat(week_frames, ignore_index=True)


def add_play_train_test_split(features: pd.DataFrame, test_size: float, random_state: int) -> pd.DataFrame:
    """Split by play."""
    if not 0.0 < test_size<1.0:
        raise ValueError("--test-size must be between 0 and 1.")
    out = features.copy()
    plays = out[["gameId","playId"]].drop_duplicates().reset_index(drop=True)
    if len(plays) < 2:
        out["split"] = "train"
        return out

    train_plays, test_plays = train_test_split( plays, test_size=test_size, random_state=random_state)
    train_plays = train_plays.assign(split="train")
    test_plays = test_plays.assign(split="test")
    split_lookup = pd.concat([train_plays, test_plays], ignore_index=True)
    return out.merge(split_lookup, on=["gameId", "playId"],how="left")


def add_week_train_test_split(features: pd.DataFrame,test_weeks:list[int]) -> pd.DataFrame:
    out = features.copy()
    if "week" not in out.columns:
        raise ValueError("Week split needs week.")
    out["split"] = np.where(out["week"].isin(set(test_weeks)),"test", "train")
    if not out["split"].eq("train").any() or not out["split"].eq("test").any():
        raise ValueError("--test-weeks needs train and test.")
    return out


def add_calibration_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    depth_bins = [-np.inf, 5.0, 10.0, 20.0, np.inf]
    depth_labels = ["short", "medium", "deep", "very_deep"]
    out["cal_depth_bucket"] = pd.cut(out["target_depth"], bins=depth_bins, labels=depth_labels).astype(str)
    out["cal_pressure_bucket"] = np.where(out["is_under_pressure"].fillna(0).astype(float).ge(0.5), "pressure", "clean")
    out["cal_bucket"] = out["cal_depth_bucket"] + "_" + out["cal_pressure_bucket"]
    return out


def bucket_probability_lookup(train_data: pd.DataFrame, target_col: str, smoothing: float) -> tuple[dict[str, float], float]:
    global_rate = float(train_data[target_col].mean()) if len(train_data) else 0.0
    grouped = train_data.groupby("cal_bucket")[target_col].agg(["sum", "count"])
    rates = (grouped["sum"] + global_rate * smoothing) / (grouped["count"] + smoothing)
    return rates.to_dict(), global_rate


def apply_bucket_calibration(raw_probability: np.ndarray, buckets: pd.Series, lookup: dict[str, float], global_rate: float, weight: float) -> np.ndarray:
    bucket_rate = buckets.map(lookup).fillna(global_rate).astype(float).to_numpy()
    calibrated = (1.0 - weight) * raw_probability + weight * bucket_rate
    return np.clip(calibrated, 0.0, 1.0)


def fit_expected_yards_models(features: pd.DataFrame, random_state: int, interception_yard_penalty: float, similar_throws: int, bayes_model_strength: float, interception_positive_weight_cap: float = 50.0) -> tuple[pd.DataFrame, dict]:
    model_data = features.loc[features["wasTargettedReceiver"].eq(1) & features["is_throw_frame"].eq(True)].copy()
    model_data = add_calibration_buckets(model_data)
    model_data = model_data.dropna(subset=FEATURE_COLUMNS +["actual_epa"])
    if model_data.empty:
        raise ValueError("No targeted throw-frame rows were available for model training.")
    x = model_data[FEATURE_COLUMNS]
    train_mask = model_data["split"].eq("train")
    test_mask = model_data["split"].eq("test")
    if not train_mask.any() or not test_mask.any():
        groups = model_data["gameId"].astype(str) + "-" + model_data["playId"].astype(str)
        train_idx, test_idx = grouped_train_valid_indices(x, groups, random_state=random_state)
        train_mask =pd.Series(False,index=model_data.index)
        test_mask = pd.Series(False,index=model_data.index)
        train_mask.iloc[train_idx]=True
        test_mask.iloc[test_idx]=True

    metrics: dict[str, float | int | str | None] = {
        "model_family": TREE_MODEL_FAMILY,
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "all_target_throw_rows": int(len(model_data)),
        "targeted_plays": int(model_data[["gameId", "playId"]].drop_duplicates().shape[0]),
        "train_plays": int(model_data.loc[train_mask, ["gameId", "playId"]].drop_duplicates().shape[0]),
        "test_plays": int(model_data.loc[test_mask, ["gameId", "playId"]].drop_duplicates().shape[0]),
        "feature_rows": int(len(features)),
        "feature_train_rows": int(features["split"].eq("train").sum()),
        "feature_test_rows": int(features["split"].eq("test").sum()),
        "eligible_train_plays": int(features.loc[features["split"].eq("train"), ["gameId", "playId"]].drop_duplicates().shape[0]),
        "eligible_test_plays": int(features.loc[features["split"].eq("test"), ["gameId", "playId"]].drop_duplicates().shape[0]),
        "training_label_frame": "actual_throw_frame_only",
        "split_unit": "play",
        "interception_yard_penalty": float(interception_yard_penalty),
        "similar_throws_k":int(similar_throws),
        "bayes_model_strength":float(bayes_model_strength)}

    y_complete = model_data["hadPassReception"].astype(int)
    completion_model = fit_classifier( x.loc[train_mask], y_complete.loc[train_mask], random_state=random_state)
    completion_lookup, completion_global = bucket_probability_lookup(model_data.loc[train_mask], "hadPassReception", COMPLETION_SMOOTHING)
    completion_valid_raw = positive_class_probability(completion_model, x.loc[test_mask])
    completion_valid_prob = apply_bucket_calibration(completion_valid_raw, model_data.loc[test_mask, "cal_bucket"], completion_lookup, completion_global, COMPLETION_CALIBRATION_WEIGHT)
    metrics["completion_brier_raw"] = float(brier_score_loss(y_complete.loc[test_mask], completion_valid_raw))
    metrics["completion_brier"] = float(brier_score_loss(y_complete.loc[test_mask], completion_valid_prob))
    metrics["completion_auc"] = (float(roc_auc_score(y_complete.loc[test_mask], completion_valid_prob))
        if y_complete.loc[test_mask].nunique() > 1
        else None)

    y_interception = model_data["is_interception"].astype(int)
    interception_model = fit_classifier(x.loc[train_mask], y_interception.loc[train_mask], random_state=random_state, positive_weight_cap=interception_positive_weight_cap)
    interception_lookup, interception_global = bucket_probability_lookup(model_data.loc[train_mask], "is_interception", INTERCEPTION_SMOOTHING)
    interception_valid_raw = positive_class_probability(interception_model, x.loc[test_mask])
    interception_valid_prob = apply_bucket_calibration(interception_valid_raw, model_data.loc[test_mask, "cal_bucket"], interception_lookup, interception_global, INTERCEPTION_CALIBRATION_WEIGHT)
    metrics["interception_brier_raw"] = float(brier_score_loss(y_interception.loc[test_mask], interception_valid_raw))
    metrics["interception_brier"] = float(brier_score_loss(y_interception.loc[test_mask], interception_valid_prob))
    metrics["interception_auc"] = (float(roc_auc_score(y_interception.loc[test_mask], interception_valid_prob))
        if y_interception.loc[test_mask].nunique() > 1
        else None)
    train_interceptions = int(y_interception.loc[train_mask].sum())
    train_non_interceptions = int(train_mask.sum() - train_interceptions)
    metrics["interception_train_positives"] = train_interceptions
    metrics["interception_train_negatives"] = train_non_interceptions
    metrics["interception_positive_weight_cap"] = float(interception_positive_weight_cap)
    metrics["probability_calibration"] = "depth_pressure_bucket_blend"
    metrics["completion_calibration_weight"] = COMPLETION_CALIBRATION_WEIGHT
    metrics["interception_calibration_weight"] = INTERCEPTION_CALIBRATION_WEIGHT

    epa_model = fit_regressor(x.loc[train_mask],model_data["actual_epa"].astype(float).loc[train_mask],random_state=random_state)
    valid_epa = epa_model.predict(x.loc[test_mask])
    metrics["epa_mae"] = float(mean_absolute_error(model_data["actual_epa"].astype(float).loc[test_mask], valid_epa))
    completed = model_data.loc[model_data["hadPassReception"].eq(1)].copy()
    completed_train = completed.loc[completed["split"].eq("train")]
    completed_test = completed.loc[completed["split"].eq("test")]
    metrics["completed_target_rows"] = int(len(completed))
    metrics["completed_train_rows"] = int(len(completed_train))
    metrics["completed_test_rows"] = int(len(completed_test))
    predicted_yards_if_completed_low = None
    predicted_yards_if_completed_high = None
    if len(completed_train) >= 20:
        yards_x_train = completed_train[FEATURE_COLUMNS]
        air_yards_model = fit_regressor(yards_x_train,completed_train["actual_air_yards"].astype(float),random_state=random_state)
        yac_model = fit_regressor(yards_x_train,completed_train["actual_yac"].astype(float),random_state=random_state)
        if len(completed_test) > 0:
            valid_air_yards = air_yards_model.predict(completed_test[FEATURE_COLUMNS])
            valid_yac = yac_model.predict(completed_test[FEATURE_COLUMNS]).clip(min=0)
            valid_yards = valid_air_yards+valid_yac
            metrics["yards_mae"]= float(mean_absolute_error(completed_test["receivingYards"].astype(float), valid_yards))
            metrics["air_yards_mae"] = float(mean_absolute_error(completed_test["actual_air_yards"].astype(float), valid_air_yards))
            metrics["yac_mae"] = float(mean_absolute_error(completed_test["actual_yac"].astype(float), valid_yac))
        else:
            metrics["yards_mae"]=None
            metrics["air_yards_mae"]=None
            metrics["yac_mae"] = None
        predicted_air_yards = air_yards_model.predict(features[FEATURE_COLUMNS])
        predicted_yac = yac_model.predict(features[FEATURE_COLUMNS]).clip(min=0)
        predicted_yards_if_completed = predicted_air_yards + predicted_yac
        if len(completed_train) >= 80:
            air_yards_low_model = fit_quantile_regressor(
                yards_x_train,
                completed_train["actual_air_yards"].astype(float),
                quantile=0.25,
                random_state=random_state,
            )
            air_yards_high_model = fit_quantile_regressor(
                yards_x_train,
                completed_train["actual_air_yards"].astype(float),
                quantile=0.75,
                random_state=random_state,
            )
            yac_low_model = fit_quantile_regressor(
                yards_x_train,
                completed_train["actual_yac"].astype(float),
                quantile=0.25,
                random_state=random_state,
            )
            yac_high_model = fit_quantile_regressor(
                yards_x_train,
                completed_train["actual_yac"].astype(float),
                quantile=0.75,
                random_state=random_state,
            )
            predicted_air_yards_low = air_yards_low_model.predict(features[FEATURE_COLUMNS])
            predicted_air_yards_high = air_yards_high_model.predict(features[FEATURE_COLUMNS])
            predicted_yac_low = yac_low_model.predict(features[FEATURE_COLUMNS]).clip(min=0)
            predicted_yac_high = yac_high_model.predict(features[FEATURE_COLUMNS]).clip(min=0)
            predicted_yards_if_completed_low = predicted_air_yards_low + predicted_yac_low
            predicted_yards_if_completed_high = predicted_air_yards_high + predicted_yac_high
            metrics["yards_range_method"] = "air_yards_plus_yac_quantile_regression_25_75"
        else:
            metrics["yards_range_method"] = "residual_fallback_due_to_small_sample"
    else:
        air_yards_model = DummyRegressor(strategy="mean")
        yac_model = DummyRegressor(strategy="mean")
        fallback = completed_train["receivingYards"].astype(float)
        if fallback.empty:
            fallback = pd.Series([0.0])
        air_fallback = completed_train["actual_air_yards"].astype(float)
        yac_fallback = completed_train["actual_yac"].astype(float)
        if air_fallback.empty:
            air_fallback = pd.Series([0.0])
        if yac_fallback.empty:
            yac_fallback = pd.Series([0.0])
        air_yards_model.fit(np.zeros((len(air_fallback), 1)), air_fallback)
        yac_model.fit(np.zeros((len(yac_fallback), 1)), yac_fallback)
        metrics["yards_mae"] = None
        metrics["air_yards_mae"] = None
        metrics["yac_mae"] = None
        metrics["yards_model"] = "dummy_air_yards_plus_yac_due_to_small_completed_sample"
        predicted_air_yards = np.repeat(float(air_fallback.mean()), len(features))
        predicted_yac = np.repeat(float(yac_fallback.mean()), len(features))
        predicted_yards_if_completed = predicted_air_yards + predicted_yac
        metrics["yards_range_method"] = "dummy_mean_due_to_small_completed_sample"

    if predicted_yards_if_completed_low is None or predicted_yards_if_completed_high is None:
        if len(completed_train):
            train_predictions = (air_yards_model.predict(completed_train[FEATURE_COLUMNS]) + yac_model.predict(completed_train[FEATURE_COLUMNS]).clip(min=0))
        else:
            train_predictions = []
        residuals = (
            completed_train["receivingYards"].astype(float).to_numpy()-np.asarray(train_predictions)
            if len(completed_train)
            else np.array([0.0])
        )
        low_offset=float(np.quantile(residuals, 0.25))
        high_offset=float(np.quantile(residuals, 0.75))
        predicted_yards_if_completed_low = predicted_yards_if_completed+low_offset
        predicted_yards_if_completed_high = predicted_yards_if_completed+high_offset

    predicted_yards_if_completed_low = np.minimum(predicted_yards_if_completed_low, predicted_yards_if_completed)
    predicted_yards_if_completed_high = np.maximum(predicted_yards_if_completed_high, predicted_yards_if_completed)
    scored = add_calibration_buckets(features)
    scored_features = scored[FEATURE_COLUMNS]
    scored["completion_probability_raw"] = positive_class_probability(completion_model, scored_features)
    scored["completion_probability"] = apply_bucket_calibration(scored["completion_probability_raw"].to_numpy(), scored["cal_bucket"], completion_lookup, completion_global, COMPLETION_CALIBRATION_WEIGHT)
    uncertainty = estimate_completion_uncertainty(scored=scored, train_throws=model_data.loc[train_mask], model_completion_probability=scored["completion_probability"].to_numpy(), k=similar_throws, model_strength=bayes_model_strength,)
    for column in uncertainty.columns:
        scored[column] = uncertainty[column].to_numpy()
    scored["interception_probability_raw"] = positive_class_probability(interception_model, scored_features)
    scored["interception_probability"] = apply_bucket_calibration(scored["interception_probability_raw"].to_numpy(), scored["cal_bucket"], interception_lookup, interception_global, INTERCEPTION_CALIBRATION_WEIGHT)
    scored["incompletion_probability"] = (1.0 - scored["completion_probability"] - scored["interception_probability"]).clip(lower=0.0)
    scored["predicted_yards_if_completed"] = predicted_yards_if_completed
    scored["predicted_air_yards_if_completed"] = predicted_air_yards
    scored["predicted_yac_if_completed"] = predicted_yac
    scored["predicted_yards_if_completed_low"] = predicted_yards_if_completed_low
    scored["predicted_yards_if_completed_high"] = predicted_yards_if_completed_high
    scored["predicted_yards_if_completed_p25"] = scored["predicted_yards_if_completed_low"]
    scored["predicted_yards_if_completed_p50"] = scored["predicted_yards_if_completed"]
    scored["predicted_yards_if_completed_p75"] = scored["predicted_yards_if_completed_high"]
    scored["predicted_yards_if_completed_iqr"] = (scored["predicted_yards_if_completed_p75"] - scored["predicted_yards_if_completed_p25"])
    scored["completion_expected_yards"] = (scored["completion_probability"] * scored["predicted_yards_if_completed"])
    scored["completion_expected_yards_low"] = (scored["completion_probability"] * scored["predicted_yards_if_completed_low"])
    scored["completion_expected_yards_high"] = (scored["completion_probability"] * scored["predicted_yards_if_completed_high"])
    scored["interception_yards_risk"] = (scored["interception_probability"] * interception_yard_penalty)
    scored["risk_adjusted_expected_yards"] = (scored["completion_probability"] * scored["predicted_yards_if_completed"] + scored["interception_yards_risk"])
    scored["risk_adjusted_expected_yards_low"] = (scored["completion_expected_yards_low"] + scored["interception_yards_risk"])
    scored["risk_adjusted_expected_yards_high"] = (scored["completion_expected_yards_high"] + scored["interception_yards_risk"])
    scored["expected_yards_p25"] = (scored["completion_probability_p25"] * scored["predicted_yards_if_completed_p25"] + scored["interception_yards_risk"])
    scored["expected_yards_p50"] = (scored["completion_probability_p50"] * scored["predicted_yards_if_completed_p50"] + scored["interception_yards_risk"])
    scored["expected_yards_p75"] = (scored["completion_probability_p75"] * scored["predicted_yards_if_completed_p75"] + scored["interception_yards_risk"])
    scored["expected_yards_iqr"] = scored["expected_yards_p75"] - scored["expected_yards_p25"]
    scored["expected_yards"] = scored["expected_yards_p50"]
    scored["predicted_epa"] = epa_model.predict(scored_features)
    return scored, metrics


def grouped_train_valid_indices(x: pd.DataFrame, groups: pd.Series, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    if groups.nunique() > 1 and len(x) >= 5:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        return next(splitter.split(x, groups=groups))
    indices = np.arange(len(x))
    if len(indices) < 2:
        return indices, indices
    train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
    return np.array(train_idx), np.array(valid_idx)




def fit_classifier(x_train: pd.DataFrame, y_train: pd.Series, random_state: int, positive_weight_cap: float = 25.0):
    if y_train.nunique() <= 1:
        model = DummyClassifier(strategy="most_frequent")
        model.fit(x_train, y_train)
        return model

    class_counts = y_train.value_counts()
    pos = int(class_counts.get(1, 0))
    neg = int(class_counts.get(0, 0))
    scale_pos_weight = 1.0
    if pos > 0 and neg > 0 and pos / (pos + neg) < 0.10:
        scale_pos_weight = min(float(positive_weight_cap), neg / pos)

    base_model = XGBClassifier(
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )
    if len(y_train) >= 60 and class_counts.min() >= 3:
        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    else:
        model = base_model
    model.fit(x_train, y_train)
    return model


def fit_regressor(
    x_train: pd.DataFrame, y_train: pd.Series, random_state: int
):
    if len(y_train) < 20:
        model = DummyRegressor(strategy="mean")
    else:
        model = XGBRegressor(
            n_estimators=350,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    model.fit(x_train, y_train)
    return model


def fit_quantile_regressor(
    x_train: pd.DataFrame, y_train: pd.Series, quantile: float, random_state: int
):
    model = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        n_estimators=350,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x_train, y_train)
    return model


def estimate_completion_uncertainty( scored: pd.DataFrame,train_throws: pd.DataFrame, model_completion_probability: np.ndarray, k: int, model_strength: float) -> pd.DataFrame:
    available_features = [
        column
        for column in SIMILARITY_FEATURE_COLUMNS
        if column in scored.columns and column in train_throws.columns
    ]
    if not available_features or train_throws.empty:
        return beta_completion_interval( model_completion_probability,effective_sample_size=np.ones(len(scored)), similar_completion_rate=model_completion_probability, model_strength=model_strength, )

    train_x = train_throws[available_features].replace([np.inf, -np.inf], np.nan)
    scored_x = scored[available_features].replace([np.inf, -np.inf], np.nan)
    medians = train_x.median(numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(medians)
    scored_x = scored_x.fillna(medians)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    scored_scaled = scaler.transform(scored_x)
    n_neighbors = max(1, min(k, len(train_throws)))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    neighbors.fit(train_scaled)
    distances, indices = neighbors.kneighbors(scored_scaled)

    bandwidth = np.nanmedian(distances[:, -1])
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = 1.0
    weights = np.exp(-distances / bandwidth)
    outcomes = train_throws["hadPassReception"].astype(float).to_numpy()
    neighbor_outcomes = outcomes[indices]
    weight_sums = weights.sum(axis=1)
    similar_completion_rate = (weights * neighbor_outcomes).sum(axis=1) / np.maximum( weight_sums, 1e-9)
    effective_sample_size = np.square(weight_sums) / np.maximum(np.square(weights).sum(axis=1), 1e-9)
    uncertainty = beta_completion_interval(model_completion_probability=model_completion_probability, effective_sample_size=effective_sample_size, similar_completion_rate=similar_completion_rate, model_strength=model_strength)
    uncertainty["similar_throw_mean_distance"] = distances.mean(axis=1)
    uncertainty["similar_throw_min_distance"] = distances[:, 0]
    return uncertainty


def beta_completion_interval(model_completion_probability: np.ndarray, effective_sample_size: np.ndarray, similar_completion_rate: np.ndarray, model_strength: float) -> pd.DataFrame:
    total_strength = np.maximum(model_strength + effective_sample_size, 1e-6)
    bayes_probability = ( model_completion_probability * model_strength + similar_completion_rate * effective_sample_size) / total_strength
    alpha = bayes_probability * total_strength + 1.0
    beta = (1.0 - bayes_probability) * total_strength + 1.0
    low = beta_distribution.ppf(0.25, alpha, beta)
    high = beta_distribution.ppf(0.75, alpha, beta)

    low = np.clip(low, 0.0, 1.0)
    high = np.clip(high, 0.0, 1.0)
    return pd.DataFrame({"completion_probability_bayes": bayes_probability, "completion_probability_low": low, "completion_probability_high": high, "completion_probability_p25": low, "completion_probability_p50": bayes_probability, "completion_probability_p75": high, "completion_probability_iqr": high - low, "completion_probability_range": high - low, "similar_throw_effective_sample_size": effective_sample_size, "similar_throw_completion_rate": similar_completion_rate})


def positive_class_probability(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x))
    return probabilities[:, classes.index(1)]


def build_play_decision_summary(scored: pd.DataFrame, acceptable_yards_gap: float, acceptable_epa_gap: float, acceptable_yards_pct: float = 0.20) -> pd.DataFrame:
    throw_frames = scored.loc[scored["is_throw_frame"].eq(True)].copy()
    if throw_frames.empty:
        return pd.DataFrame()

    group_cols = ["gameId", "playId"]
    best_epa_idx = throw_frames.groupby(group_cols)["predicted_epa"].idxmax()
    actual = throw_frames.loc[throw_frames["wasTargettedReceiver"].eq(1)].copy()
    if actual.empty:
        return pd.DataFrame()

    actual = actual[group_cols + ["week", "frameId", "split", "nflId", "displayName", "routeRan", "seconds_since_snap", "expected_yards", "expected_yards_p25", "expected_yards_p50", "expected_yards_p75", "risk_adjusted_expected_yards_low", "risk_adjusted_expected_yards_high", "completion_expected_yards", "interception_yards_risk", "predicted_epa", "completion_probability", "completion_probability_bayes", "completion_probability_low", "completion_probability_high", "completion_probability_p25", "completion_probability_p50", "completion_probability_p75", "completion_probability_range", "similar_throw_effective_sample_size", "similar_throw_completion_rate", "interception_probability", "receivingYards", "actual_epa", "hadPassReception", "is_interception"]].rename( columns={ "frameId": "throw_frame_id", "nflId": "actual_target_nflId", "displayName": "actual_target_name", "routeRan": "actual_target_route", "seconds_since_snap": "actual_throw_seconds_since_snap", "expected_yards": "actual_target_expected_yards", "expected_yards_p25": "actual_target_expected_yards_p25", "expected_yards_p50": "actual_target_expected_yards_p50", "expected_yards_p75": "actual_target_expected_yards_p75", "risk_adjusted_expected_yards_low": "actual_target_expected_yards_low", "risk_adjusted_expected_yards_high": "actual_target_expected_yards_high", "completion_expected_yards": "actual_target_completion_expected_yards", "interception_yards_risk": "actual_target_interception_yards_risk", "predicted_epa": "actual_target_predicted_epa", "completion_probability": "actual_target_completion_probability", "completion_probability_bayes": "actual_target_completion_probability_bayes", "completion_probability_low": "actual_target_completion_probability_low", "completion_probability_high": "actual_target_completion_probability_high", "completion_probability_p25": "actual_target_completion_probability_p25", "completion_probability_p50": "actual_target_completion_probability_p50", "completion_probability_p75": "actual_target_completion_probability_p75", "completion_probability_range": "actual_target_completion_probability_range", "similar_throw_effective_sample_size": "actual_target_similar_throw_effective_sample_size", "similar_throw_completion_rate": "actual_target_similar_throw_completion_rate", "interception_probability": "actual_target_interception_probability"})

    summary = actual.copy()

    for percentile in ("p25", "p50", "p75"):
        value_col = f"expected_yards_{percentile}"
        best_idx = throw_frames.groupby(group_cols)[value_col].idxmax()
        best = throw_frames.loc[ best_idx, group_cols + [ "nflId","displayName","frameId", "seconds_since_snap", value_col, "interception_yards_risk","predicted_epa"]].rename(columns={ "nflId": f"best_yards_option_{percentile}_nflId","displayName": f"best_yards_option_{percentile}_name","frameId": f"best_yards_option_{percentile}_frame_id","seconds_since_snap": f"best_yards_option_{percentile}_seconds_since_snap",value_col: f"best_available_expected_yards_{percentile}","interception_yards_risk": f"best_available_interception_yards_risk_{percentile}", "predicted_epa": f"best_yards_option_{percentile}_predicted_epa"})
        summary = summary.merge(best, on=group_cols, how="left")
        summary[f"missed_expected_yards_{percentile}"] =(summary[f"best_available_expected_yards_{percentile}"]-summary[f"actual_target_expected_yards_{percentile}"])
        summary[f"threw_to_best_yards_option_{percentile}"]= summary["actual_target_nflId"].eq(summary[f"best_yards_option_{percentile}_nflId"])
        summary[f"acceptable_yards_gap_{percentile}"] = np.maximum(acceptable_yards_gap, acceptable_yards_pct * summary[f"best_available_expected_yards_{percentile}"].clip(lower=0))
        summary[f"acceptable_yards_choice_{percentile}"] = summary[f"missed_expected_yards_{percentile}"].le(summary[f"acceptable_yards_gap_{percentile}"])

    pre_throw_frames = scored.merge(actual[group_cols+["throw_frame_id"]], on=group_cols, how="inner")
    pre_throw_frames = pre_throw_frames.loc[ pre_throw_frames["frameId"].le(pre_throw_frames["throw_frame_id"])].copy()

    for percentile in ("p25", "p50", "p75"):
        value_col = f"expected_yards_{percentile}"
        timing_idx = pre_throw_frames.groupby(group_cols)[value_col].idxmax()
        timing = pre_throw_frames.loc[timing_idx, group_cols+["nflId","displayName","frameId","seconds_since_snap",value_col,"predicted_epa"]].rename(columns={ "nflId": f"best_timing_option_{percentile}_nflId","displayName": f"best_timing_option_{percentile}_name","frameId": f"best_timing_option_{percentile}_frame_id","seconds_since_snap": f"best_timing_option_{percentile}_seconds_since_snap", value_col: f"best_timing_expected_yards_{percentile}", "predicted_epa": f"best_timing_predicted_epa_{percentile}"})
        summary = summary.merge(timing, on=group_cols, how="left")
        summary[f"timing_loss_expected_yards_{percentile}"] =(summary[f"best_timing_expected_yards_{percentile}"] - summary[f"actual_target_expected_yards_{percentile}"])
        summary[f"timing_seconds_from_actual_throw_{percentile}"]= (summary[f"best_timing_option_{percentile}_seconds_since_snap"] - summary["actual_throw_seconds_since_snap"])
        summary[f"right_receiver_right_time_{percentile}"]= (summary["actual_target_nflId"].eq(summary[f"best_timing_option_{percentile}_nflId"]) & summary["throw_frame_id"].eq(summary[f"best_timing_option_{percentile}_frame_id"]))
        summary[f"acceptable_timing_gap_{percentile}"] = np.maximum(acceptable_yards_gap, acceptable_yards_pct * summary[f"best_timing_expected_yards_{percentile}"].clip(lower=0))
        summary[f"acceptable_timing_choice_{percentile}"]= summary[f"timing_loss_expected_yards_{percentile}"].le(summary[f"acceptable_timing_gap_{percentile}"])
    best_epa = throw_frames.loc[best_epa_idx, group_cols+["nflId", "displayName", "expected_yards", "predicted_epa"]].rename(columns={ "nflId": "best_epa_option_nflId", "displayName": "best_epa_option_name",  "expected_yards": "best_epa_option_expected_yards",  "predicted_epa": "best_available_predicted_epa" })

    summary = summary.merge(best_epa, on=group_cols, how="left")
    summary["best_yards_option_nflId"] = summary["best_yards_option_p50_nflId"]
    summary["best_yards_option_name"] = summary["best_yards_option_p50_name"]
    summary["best_available_expected_yards"] = summary["best_available_expected_yards_p50"]
    summary["best_available_expected_yards_low"] = summary["best_available_expected_yards_p25"]
    summary["best_available_expected_yards_high"] = summary["best_available_expected_yards_p75"]
    summary["best_available_interception_yards_risk"] = summary[ "best_available_interception_yards_risk_p50"]
    summary["best_yards_option_predicted_epa"] = summary["best_yards_option_p50_predicted_epa"]
    summary["missed_expected_yards"] = (summary["best_available_expected_yards"] - summary["actual_target_expected_yards"])
    summary["missed_predicted_epa"] = (summary["best_available_predicted_epa"] - summary["actual_target_predicted_epa"])
    summary["acceptable_epa_choice"] = summary["missed_predicted_epa"].le(acceptable_epa_gap)
    summary["result_yards_over_expected"] = (summary["receivingYards"] - summary["actual_target_expected_yards_p50"])
    summary["result_epa_over_predicted"] = (summary["actual_epa"] - summary["actual_target_predicted_epa"])
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plays = read_csv(args.data_dir, "plays.csv")
    player_play = read_csv(args.data_dir, "player_play.csv")
    players = read_csv(args.data_dir, "players.csv")
    tracking = load_tracking_weeks(args.data_dir, args.weeks)

    features = build_feature_table(tracking=tracking, plays=plays, player_play=player_play, players=players, min_play_seconds=args.min_play_seconds, max_plays=args.max_plays)
    if features.empty:
        raise ValueError("No eligible passing plays were found.")

    features = features.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    if args.test_weeks:
        features = add_week_train_test_split(features, args.test_weeks)
    else:
        features = add_play_train_test_split(features, test_size=args.test_size, random_state=args.random_state)
    scored, metrics = fit_expected_yards_models( features, random_state=args.random_state, interception_yard_penalty=args.interception_yard_penalty, similar_throws=args.similar_throws, bayes_model_strength=args.bayes_model_strength, interception_positive_weight_cap=args.interception_positive_weight_cap)

    feature_path = args.output_dir / "receiver_opportunity_features.csv"
    scored_path = args.output_dir / "receiver_expected_yards.csv"
    summary_path = args.output_dir / "play_decision_summary.csv"
    metrics_path = args.output_dir / "model_metrics.json"

    feature_columns_to_write = [ "gameId", "playId", "week","frameId", "nflId", "displayName", "routeRan", "split", "end_reason","play_duration_seconds", "is_throw_frame", "passResult", "wasTargettedReceiver", "hadPassReception", "is_interception", "receivingYards","actual_air_yards","actual_yac","actual_epa",*FEATURE_COLUMNS,]
    score_columns_to_write = [ *feature_columns_to_write, "cal_depth_bucket", "cal_pressure_bucket", "cal_bucket", "completion_probability_raw", "completion_probability", "completion_probability_bayes", "completion_probability_low", "completion_probability_high", "completion_probability_p25", "completion_probability_p50", "completion_probability_p75", "completion_probability_iqr", "completion_probability_range", "similar_throw_effective_sample_size", "similar_throw_completion_rate", "similar_throw_mean_distance","similar_throw_min_distance","interception_probability_raw", "interception_probability", "incompletion_probability", "predicted_yards_if_completed", "predicted_air_yards_if_completed", "predicted_yac_if_completed", "predicted_yards_if_completed_low", "predicted_yards_if_completed_high", "predicted_yards_if_completed_p25", "predicted_yards_if_completed_p50","predicted_yards_if_completed_p75", "predicted_yards_if_completed_iqr", "completion_expected_yards", "completion_expected_yards_low", "completion_expected_yards_high", "interception_yards_risk", "risk_adjusted_expected_yards", "risk_adjusted_expected_yards_low", "risk_adjusted_expected_yards_high", "expected_yards_p25", "expected_yards_p50", "expected_yards_p75", "expected_yards_iqr","expected_yards","predicted_epa"]

    features[feature_columns_to_write].to_csv(feature_path, index=False)
    scored[score_columns_to_write].to_csv(scored_path, index=False)
    play_summary = build_play_decision_summary( scored, acceptable_yards_gap=args.acceptable_yards_gap, acceptable_epa_gap=args.acceptable_epa_gap, acceptable_yards_pct=args.acceptable_yards_pct,)
    play_summary.to_csv(summary_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Wrote {len(features):,} receiver-frame feature rows to {feature_path}")
    print(f"Wrote {len(scored):,} scored receiver-frame rows to {scored_path}")
    print(f"Wrote {len(play_summary):,} play decision rows to {summary_path}")
    print(f"Wrote model metrics to {metrics_path}")


if __name__ == "__main__":
    main()
