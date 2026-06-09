"""Graph passing model."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


FIELD_LENGTH = 120.0
FIELD_WIDTH = 53.3
FRAMES_PER_SECOND = 10.0
PASS_EVENTS = {"pass_forward", "pass_shovel"}
ROUTE_TYPES = ["ANGLE", "CORNER", "CROSS", "FLAT", "GO", "HITCH", "IN", "OUT", "POST", "SCREEN", "SLANT", "WHEEL"]
ROUTE_FEATURES = [f"route_{route.lower()}" for route in ROUTE_TYPES] + ["route_other"]
GRAPH_CONTEXT_NODE_FEATURES = ["qb_facing_target", "target_depth", "target_distance", "nearest_rusher_distance", "pressure_score",
    "throwing_lane_defender_count", "throwing_lane_closest_distance", "catch_point_nearest_defender_distance"]
NODE_FEATURES = ["x_norm", "y_norm", "s", "a", "vx_norm", "vy_norm", "is_offense", "is_defense", "is_qb", "is_target", "is_route_runner", "distance_to_qb", "distance_to_target",
    "x_from_qb", "y_from_qb", "x_from_target", "y_from_target"] + ROUTE_FEATURES + GRAPH_CONTEXT_NODE_FEATURES
CONTEXT_FEATURES = [ "down", "yardsToGo", "absoluteYardlineNumber", "seconds_to_throw"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser( description="Train a PyTorch spatiotemporal graph transformer on passing plays.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("graph_transformer_output"))
    parser.add_argument("--weeks", nargs="+", type=int, default=[1])
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-players", type=int, default=22)
    parser.add_argument("--max-plays", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--yard-loss-weight", type=float, default=0.15)
    parser.add_argument("--air-yards-loss-weight", type=float, default=0.03)
    parser.add_argument("--yac-loss-weight", type=float, default=0.03)
    parser.add_argument("--interception-loss-weight", type=float, default=0.5)
    parser.add_argument("--interception-pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--prediction-task", choices=["all", "outcome", "air_yards", "yac", "total_yards"], default="all")
    parser.add_argument("--architecture", choices=["transformer", "gatv2_pna"], default="gatv2_pna")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["train", "infer"], default="train")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--inference-output", type=Path, default=None)
    return parser.parse_args()


def read_csv(data_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    path = data_dir/name
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_csv(path, **kwargs)


def normalize_tracking(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    going_right = out["playDirection"].eq("right")
    out["x_norm"] = np.where(going_right, out["x"], FIELD_LENGTH - out["x"])
    out["y_norm"] = np.where(going_right, out["y"], FIELD_WIDTH - out["y"])

    direction = np.deg2rad(out["dir"].fillna(0.0))
    vx = out["s"].fillna(0.0) * np.sin(direction)
    vy = out["s"].fillna(0.0) * np.cos(direction)
    out["vx_norm"] = np.where(going_right, vx, -vx)
    out["vy_norm"] = np.where(going_right, vy, -vy)
    orientation = np.deg2rad(out["o"].fillna(out["dir"]).fillna(0.0))
    ox = np.sin(orientation)
    oy = np.cos(orientation)
    out["orientation_x_norm"] = np.where(going_right, ox, -ox)
    out["orientation_y_norm"] = np.where(going_right, oy, -oy)
    return out


def load_tracking(data_dir: Path, weeks: Iterable[int]) -> pd.DataFrame:
    usecols = ["gameId", "playId", "nflId", "displayName", "frameId", "club", "playDirection", "x", "y", "s", "a", "dir", "o", "event"]
    frames = [read_csv(data_dir, f"tracking_week_{week}.csv", usecols=usecols) for week in weeks]
    return pd.concat(frames, ignore_index=True)


def first_event_frame(play_tracking: pd.DataFrame, events: set[str]) -> int | None:
    frames = play_tracking.loc[play_tracking["event"].isin(events),"frameId"]
    if frames.empty:
        return None
    return int(frames.min())


def add_route_features(frame: pd.DataFrame) -> pd.DataFrame:
    route = frame.get("routeRan", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.replace(" ", "_")
    for route_type in ROUTE_TYPES:
        frame[f"route_{route_type.lower()}"] = (frame["is_route_runner"].eq(1) & route.eq(route_type)).astype(float)
    frame["route_other"] = (frame["is_route_runner"].eq(1) & ~route.isin(ROUTE_TYPES)).astype(float)
    return frame


def line_x_for_frame(frame: pd.DataFrame) -> float:
    line_x = float(frame["absoluteYardlineNumber"].iloc[0]) if "absoluteYardlineNumber" in frame.columns else 0.0
    direction = str(frame["playDirection"].iloc[0]) if "playDirection" in frame.columns and len(frame) else "right"
    if direction == "left":
        line_x = FIELD_LENGTH - line_x
    return line_x


def add_graph_context_features(frame: pd.DataFrame, target_x: float, target_y: float, qb_x: float, qb_y: float, qb_row: pd.DataFrame) -> pd.DataFrame:
    target_dx = float(target_x - qb_x)
    target_dy = float(target_y - qb_y)
    target_distance = float(np.hypot(target_dx, target_dy))
    if qb_row.empty:
        qb_facing_target = 0.0
    else:
        ox = float(qb_row["orientation_x_norm"].iloc[0])
        oy = float(qb_row["orientation_y_norm"].iloc[0])
        qb_facing_target = float((ox * target_dx + oy * target_dy) / max(target_distance, 1e-6))

    defenders = frame.loc[frame["is_defense"].eq(1)]
    if defenders.empty:
        nearest_rusher_distance = 20.0
        lane_count = 0.0
        lane_closest = 10.0
        catch_point_nearest = 20.0
    else:
        defender_x = defenders["x_norm"].to_numpy(dtype=float)
        defender_y = defenders["y_norm"].to_numpy(dtype=float)
        qb_distances = np.hypot(defender_x - qb_x, defender_y - qb_y)
        nearest_rusher_distance = float(np.min(qb_distances))

        target_distances = np.hypot(defender_x - target_x, defender_y - target_y)
        catch_point_nearest = float(np.min(target_distances))

        lane_len_sq = max(target_distance * target_distance, 1e-6)
        progress = ((defender_x - qb_x) * target_dx + (defender_y - qb_y) * target_dy) / lane_len_sq
        proj_x = qb_x + progress * target_dx
        proj_y = qb_y + progress * target_dy
        lane_distances = np.hypot(defender_x - proj_x, defender_y - proj_y)
        lane_width = min(5.0, max(2.0, 1.5 + target_distance / 25.0))
        in_lane = (progress >= 0.0) & (progress <= 1.0) & (lane_distances <= lane_width)
        lane_count = float(in_lane.sum())
        lane_closest = float(lane_distances[in_lane].min()) if in_lane.any() else 10.0

    frame["qb_facing_target"] = qb_facing_target
    frame["target_depth"] = float(target_x - line_x_for_frame(frame))
    frame["target_distance"] = target_distance
    frame["nearest_rusher_distance"] = nearest_rusher_distance
    frame["pressure_score"] = float(np.clip((5.0 - nearest_rusher_distance) / 5.0, 0.0, 1.0))
    frame["throwing_lane_defender_count"] = lane_count
    frame["throwing_lane_closest_distance"] = lane_closest
    frame["catch_point_nearest_defender_distance"] = catch_point_nearest
    return add_route_features(frame)


@dataclass
class SampleIndex:
    game_id: int
    play_id: int
    target_nfl_id: int
    pass_frame: int
    split: str
    context: tuple[float, ...]
    completion: float
    interception: float
    air_yards: float
    yac: float


@dataclass
class OpportunityIndex:
    game_id: int
    play_id: int
    target_nfl_id: int
    pass_frame: int
    display_name: str = ""
    context: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    was_actual_target: float = 0.0
    completion: float = 0.0
    interception: float = 0.0
    air_yards: float = 0.0
    yac: float = 0.0


class PassingGraphDataset(Dataset):
    def __init__( self, tracking: pd.DataFrame, samples: list[SampleIndex | OpportunityIndex], node_scaler: StandardScaler, context_scaler: StandardScaler, sequence_length: int, max_players: int) -> None:
        self.tracking = tracking
        self.samples = samples
        self.node_scaler = node_scaler
        self.context_scaler = context_scaler
        self.sequence_length = sequence_length
        self.max_players = max_players
        self.play_groups = {key: group.sort_values(["frameId", "nflId"]).copy() for key, group in tracking.groupby(["gameId", "playId"], sort=False)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        play = self.play_groups[(sample.game_id, sample.play_id)]
        frame_ids = sorted(play["frameId"].unique())
        selected_frames = [frame for frame in frame_ids if frame <= sample.pass_frame]
        selected_frames = selected_frames[-self.sequence_length :]
        if len(selected_frames) < self.sequence_length:
            selected_frames = [selected_frames[0]] * (self.sequence_length - len(selected_frames)) + selected_frames

        frame_tensors =[]
        masks =[]
        target_indices =[]
        for frame_id in selected_frames:
            frame = play.loc[play["frameId"].eq(frame_id)].copy()
            target_row = frame.loc[frame["nflId"].eq(sample.target_nfl_id)]
            qb_row = frame.loc[frame["is_qb"].eq(1)]
            if target_row.empty:
                target_x = frame["target_x_fallback"].iloc[0]
                target_y = frame["target_y_fallback"].iloc[0]
            else:
                target_x = target_row["x_norm"].iloc[0]
                target_y = target_row["y_norm"].iloc[0]
            if qb_row.empty:
                qb_x = frame["x_norm"].mean()
                qb_y = frame["y_norm"].mean()
            else:
                qb_x = qb_row["x_norm"].iloc[0]
                qb_y = qb_row["y_norm"].iloc[0]

            frame["distance_to_qb"] = np.hypot(frame["x_norm"] - qb_x, frame["y_norm"] - qb_y)
            frame["distance_to_target"] = np.hypot(frame["x_norm"] - target_x, frame["y_norm"] - target_y)
            frame["x_from_qb"]=frame["x_norm"]- qb_x
            frame["y_from_qb"] = frame["y_norm"]- qb_y
            frame["x_from_target"] =frame["x_norm"]- target_x
            frame["y_from_target"]= frame["y_norm"]- target_y
            frame["is_target"] = frame["nflId"].eq(sample.target_nfl_id).astype(float)
            frame = add_graph_context_features(frame, target_x, target_y, qb_x, qb_y, qb_row)

            frame = prioritize_players(frame, sample.target_nfl_id, self.max_players)
            target_positions = np.flatnonzero(frame["nflId"].to_numpy() == sample.target_nfl_id)
            target_idx = int(target_positions[0]) if len(target_positions) else 0

            features = frame[NODE_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            features = self.node_scaler.transform(features)
            node_tensor = np.zeros((self.max_players, len(NODE_FEATURES)), dtype=np.float32)
            mask = np.zeros(self.max_players, dtype=bool)
            node_count = min(len(features), self.max_players)
            node_tensor[:node_count] = features[:node_count]
            mask[:node_count] = True

            frame_tensors.append(node_tensor)
            masks.append(mask)
            target_indices.append(target_idx)

        context = np.array(sample.context, dtype=np.float32).reshape(1, -1)
        context = self.context_scaler.transform(context).astype(np.float32).reshape(-1)

        return {"nodes": torch.tensor(np.stack(frame_tensors), dtype=torch.float32), "context": torch.tensor(context, dtype=torch.float32), "mask": torch.tensor(np.stack(masks), dtype=torch.bool), "target_index": torch.tensor(target_indices, dtype=torch.long), "gameId": torch.tensor(sample.game_id, dtype=torch.long), "playId": torch.tensor(sample.play_id, dtype=torch.long), "nflId": torch.tensor(sample.target_nfl_id, dtype=torch.long), "pass_frame": torch.tensor(sample.pass_frame, dtype=torch.long), "completion": torch.tensor(sample.completion, dtype=torch.float32), "interception": torch.tensor(sample.interception, dtype=torch.float32), "air_yards": torch.tensor(sample.air_yards, dtype=torch.float32), "yac": torch.tensor(sample.yac, dtype=torch.float32)}


def prioritize_players(frame: pd.DataFrame, target_nfl_id: int, max_players: int) -> pd.DataFrame:
    target = frame.loc[frame["nflId"].eq(target_nfl_id)]
    if target.empty:
        frame["priority_distance"] = frame["distance_to_qb"]
    else:
        tx = target["x_norm"].iloc[0]
        ty = target["y_norm"].iloc[0]
        frame["priority_distance"] = np.minimum(frame["distance_to_qb"], np.hypot(frame["x_norm"] - tx, frame["y_norm"] - ty))
    frame["priority_rank"] = (frame["is_target"] * -100 + frame["is_qb"] * -50 + frame["is_route_runner"] * -10 + frame["priority_distance"])
    return frame.sort_values("priority_rank").head(max_players).copy()


class SpatiotemporalGraphTransformer(nn.Module):
    def __init__(self, node_feature_dim: int, context_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, max_players: int) -> None:
        super().__init__()
        self.node_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(context_feature_dim),
            nn.Linear(context_feature_dim, hidden_dim),
            nn.GELU(),
        )
        graph_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.graph_encoder = nn.TransformerEncoder(graph_layer, num_layers=num_layers)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        self.player_embedding = nn.Parameter(torch.zeros(1, max_players, hidden_dim))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(
        self,
        nodes: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor,
        target_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, sequence_length, max_players, _ = nodes.shape
        x = self.node_projection(nodes)
        x = x + self.player_embedding[:, :max_players, :].unsqueeze(1)
        x = x.reshape(batch_size * sequence_length, max_players, -1)
        graph_padding_mask = ~mask.reshape(batch_size * sequence_length, max_players)
        graph_encoded = self.graph_encoder(x, src_key_padding_mask=graph_padding_mask)
        graph_encoded = graph_encoded.reshape(batch_size, sequence_length, max_players, -1)

        gather_index = target_index[:, :, None, None].expand(
            batch_size, sequence_length, 1, graph_encoded.shape[-1]
        )
        target_sequence = graph_encoded.gather(2, gather_index).squeeze(2)
        temporal_encoded = self.temporal_encoder(target_sequence)
        final_state = temporal_encoded[:, -1, :] + self.context_projection(context)
        raw = self.output(final_state)
        return {
            "completion_logit": raw[:, 0],
            "interception_logit": raw[:, 1],
            "air_yards": raw[:, 2],
            "yac": torch.relu(raw[:, 3]),
            "total_yards": raw[:, 4],
        }


class GATv2Layer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_players, _ = x.shape
        q = self.q(x).view(batch_size, max_players, self.num_heads, self.head_dim)
        k = self.k(x).view(batch_size, max_players, self.num_heads, self.head_dim)
        v = self.v(x).view(batch_size, max_players, self.num_heads, self.head_dim)
        pair = torch.tanh(q[:, :, None, :, :] + k[:, None, :, :, :])
        score = (pair * self.attn[None, None, None, :, :]).sum(-1)
        score = score.permute(0, 3, 1, 2) / np.sqrt(max(self.head_dim, 1))
        score = score.masked_fill(~mask[:, None, None, :], -1e9)
        weights = torch.softmax(score, dim=-1)
        v = v.permute(0, 2, 1, 3)
        msg = torch.matmul(weights, v).permute(0, 2, 1, 3).reshape(batch_size, max_players, -1)
        x = self.norm(x + self.out(msg))
        x = x + self.ff(x)
        return x


class GATv2PnaPassingModel(nn.Module):
    def __init__(self, node_feature_dim: int, context_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, max_players: int) -> None:
        super().__init__()
        self.max_players = max_players
        self.node_feature_dim = node_feature_dim
        self.rotate_columns = [NODE_FEATURES.index(name) for name in ["x_norm", "y_norm", "vx_norm", "vy_norm", "x_from_qb", "y_from_qb", "x_from_target", "y_from_target"] if name in NODE_FEATURES]
        self.node_projection = nn.Linear(node_feature_dim, hidden_dim)
        self.context_projection = nn.Sequential(nn.LayerNorm(context_feature_dim), nn.Linear(context_feature_dim, hidden_dim), nn.GELU())
        self.layers = nn.ModuleList([GATv2Layer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.temporal = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        pna_dim = hidden_dim * 4
        head_in = hidden_dim * 3 + pna_dim
        self.completion_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.interception_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.air_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.yac_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.total_head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def rotate_view(self, nodes: torch.Tensor) -> torch.Tensor:
        rotated = nodes.clone()
        if self.rotate_columns:
            rotated[..., self.rotate_columns] = -rotated[..., self.rotate_columns]
        return rotated

    def encode_view(self, nodes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_players, _ = nodes.shape
        x = self.node_projection(nodes).reshape(batch_size * sequence_length, max_players, -1)
        frame_mask = mask.reshape(batch_size * sequence_length, max_players)
        for layer in self.layers:
            x = layer(x, frame_mask)
        return x.reshape(batch_size, sequence_length, max_players, -1)

    def pna_pool(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask[:, :, None]
        x_zero = x.masked_fill(~valid, 0.0)
        count = valid.sum(1).clamp(min=1)
        mean = x_zero.sum(1) / count
        x_min = x.masked_fill(~valid, 1e9).min(1).values
        x_max = x.masked_fill(~valid, -1e9).max(1).values
        var = ((x_zero - mean[:, None, :]).pow(2).masked_fill(~valid, 0.0).sum(1) / count).sqrt()
        return torch.cat([mean, x_max, x_min, var], dim=-1)

    def forward(self, nodes: torch.Tensor, context: torch.Tensor, mask: torch.Tensor, target_index: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, sequence_length, max_players, _ = nodes.shape
        encoded = 0.5 * (self.encode_view(nodes, mask) + self.encode_view(self.rotate_view(nodes), mask))
        gather_index = target_index[:, :, None, None].expand(batch_size, sequence_length, 1, encoded.shape[-1])
        target_sequence = encoded.gather(2, gather_index).squeeze(2)
        target_temporal, _ = self.temporal(target_sequence)
        target_temporal = target_temporal[:, -1, :]
        final_frame = encoded[:, -1, :, :]
        final_mask = mask[:, -1, :]
        target_final = target_sequence[:, -1, :]
        global_context = (final_frame.masked_fill(~final_mask[:, :, None], 0.0).sum(1) / final_mask.sum(1, keepdim=True).clamp(min=1))
        pna = self.pna_pool(final_frame, final_mask)
        ctx = self.context_projection(context)
        z = torch.cat([target_temporal, target_final, global_context + ctx, pna], dim=-1)
        return {
            "completion_logit": self.completion_head(z).squeeze(-1),
            "interception_logit": self.interception_head(z).squeeze(-1),
            "air_yards": self.air_head(z).squeeze(-1),
            "yac": torch.relu(self.yac_head(z).squeeze(-1)),
            "total_yards": self.total_head(z).squeeze(-1),
        }


def make_graph_model(architecture: str, node_feature_dim: int, context_feature_dim: int, hidden_dim: int, num_heads: int, num_layers: int, max_players: int) -> nn.Module:
    if architecture == "gatv2_pna":
        return GATv2PnaPassingModel(node_feature_dim, context_feature_dim, hidden_dim, num_heads, num_layers, max_players)
    return SpatiotemporalGraphTransformer(node_feature_dim, context_feature_dim, hidden_dim, num_heads, num_layers, max_players)


def build_samples( tracking: pd.DataFrame,plays: pd.DataFrame, player_play: pd.DataFrame,players: pd.DataFrame,max_plays: int | None, test_size: float, random_state: int) -> tuple[pd.DataFrame, list[SampleIndex]]:
    plays = plays.loc[plays["isDropback"].eq(True)].copy()
    targets = player_play.loc[player_play["wasTargettedReceiver"].eq(1)].copy()
    route_runners = player_play.loc[player_play["wasRunningRoute"].eq(1), ["gameId", "playId", "nflId", "routeRan"]]

    tracking = tracking.merge(players[["nflId", "position"]], on="nflId", how="left")
    tracking = normalize_tracking(tracking)
    tracking = tracking.merge( plays[[ "gameId", "playId", "possessionTeam", "defensiveTeam", "down", "yardsToGo", "absoluteYardlineNumber"]], on=["gameId", "playId"],how="inner",)
    tracking = tracking.merge(route_runners.assign(is_route_runner=1), on=["gameId", "playId", "nflId"], how="left")
    tracking["is_route_runner"] = tracking["is_route_runner"].fillna(0).astype(float)
    tracking["routeRan"] = tracking["routeRan"].fillna("")
    tracking["is_qb"] = tracking["position"].eq("QB").astype(float)
    tracking["is_offense"] = tracking["club"].eq(tracking["possessionTeam"]).astype(float)
    tracking["is_defense"] = tracking["club"].eq(tracking["defensiveTeam"]).astype(float)
    tracking["target_x_fallback"] = tracking.groupby(["gameId", "playId"])["x_norm"].transform("mean")
    tracking["target_y_fallback"] = tracking.groupby(["gameId", "playId"])["y_norm"].transform("mean")

    play_keys = tracking[["gameId", "playId"]].drop_duplicates()
    if max_plays is not None:
        play_keys = play_keys.head(max_plays)
        tracking = tracking.merge(play_keys, on=["gameId", "playId"], how="inner")

    split_keys = play_keys.copy()
    train_keys, test_keys = train_test_split( split_keys, test_size=test_size, random_state=random_state)
    split_lookup = pd.concat([train_keys.assign(split="train"), test_keys.assign(split="test")], ignore_index=True)

    play_lookup = plays.set_index(["gameId","playId"])
    target_lookup = targets.set_index(["gameId","playId"])
    samples: list[SampleIndex] =[]
    for (game_id, play_id), play_tracking in tracking.groupby(["gameId", "playId"], sort=False):
        pass_frame = first_event_frame(play_tracking, PASS_EVENTS)
        if pass_frame is None:
            continue
        key = (game_id, play_id)
        if key not in target_lookup.index or key not in play_lookup.index:
            continue
        target_row = target_lookup.loc[key]
        if isinstance(target_row,pd.DataFrame):
            target_row = target_row.iloc[0]
        split = split_lookup.loc[split_lookup["gameId"].eq(game_id) & split_lookup["playId"].eq(play_id), "split"]
        if split.empty:
            continue
        yac = max(float(target_row["yardageGainedAfterTheCatch"]), 0.0)
        receiving_yards = float(target_row["receivingYards"])
        play_row = play_lookup.loc[key]
        context = (float(play_row.get("down", 0.0)),float(play_row.get("yardsToGo", 0.0)), float(play_row.get("absoluteYardlineNumber", 0.0)), float((pass_frame - first_event_frame(play_tracking, {"ball_snap"})) / FRAMES_PER_SECOND) if first_event_frame(play_tracking, {"ball_snap"}) is not None else 0.0,)
        samples.append( SampleIndex(game_id=int(game_id), play_id=int(play_id), target_nfl_id=int(target_row["nflId"]), pass_frame=int(pass_frame), split=str(split.iloc[0]), context=context, completion=float(target_row["hadPassReception"]), interception=float(play_lookup.loc[key, "passResult"] == "IN"), air_yards=receiving_yards - yac, yac=yac))
    return tracking, samples


def build_inference_opportunities(tracking: pd.DataFrame, plays: pd.DataFrame, player_play: pd.DataFrame, players: pd.DataFrame, max_plays: int | None) -> tuple[pd.DataFrame, list[OpportunityIndex]]:
    plays = plays.loc[plays["isDropback"].eq(True)].copy()
    route_runners = player_play.loc[ player_play["wasRunningRoute"].eq(1), ["gameId", "playId", "nflId", "routeRan", "wasTargettedReceiver", "hadPassReception", "receivingYards", "yardageGainedAfterTheCatch"]].copy()

    tracking = tracking.merge(players[["nflId", "position"]], on="nflId", how="left")
    tracking = normalize_tracking(tracking)
    tracking = tracking.merge( plays[["gameId", "playId", "possessionTeam", "defensiveTeam", "passResult", "down", "yardsToGo", "absoluteYardlineNumber"]], on=["gameId", "playId"],how="inner")
    tracking = tracking.merge(route_runners[["gameId", "playId", "nflId", "routeRan"]].assign(is_route_runner=1), on=["gameId", "playId", "nflId"], how="left")
    tracking["is_route_runner"] = tracking["is_route_runner"].fillna(0).astype(float)
    tracking["routeRan"] = tracking["routeRan"].fillna("")
    tracking["is_qb"] = tracking["position"].eq("QB").astype(float)
    tracking["is_offense"] = tracking["club"].eq(tracking["possessionTeam"]).astype(float)
    tracking["is_defense"] = tracking["club"].eq(tracking["defensiveTeam"]).astype(float)
    tracking["target_x_fallback"] = tracking.groupby(["gameId", "playId"])["x_norm"].transform("mean")
    tracking["target_y_fallback"] = tracking.groupby(["gameId", "playId"])["y_norm"].transform("mean")

    play_keys = tracking[["gameId", "playId"]].drop_duplicates()
    if max_plays is not None:
        play_keys = play_keys.head(max_plays)
        tracking = tracking.merge(play_keys, on=["gameId", "playId"], how="inner")

    play_lookup = plays.set_index(["gameId", "playId"])
    route_lookup = route_runners.groupby(["gameId", "playId"], sort=False)
    display_lookup =(tracking[["gameId", "playId", "nflId", "displayName"]].dropna(subset=["nflId"]).drop_duplicates(["gameId", "playId", "nflId"]))
    display_lookup = display_lookup.set_index(["gameId", "playId", "nflId"])["displayName"]

    opportunities: list[OpportunityIndex] = []
    for (game_id, play_id), play_tracking in tracking.groupby(["gameId", "playId"], sort=False):
        pass_frame =first_event_frame(play_tracking, PASS_EVENTS)
        key = (game_id,play_id)
        if pass_frame is None or key not in play_lookup.index or key not in route_lookup.groups:
            continue
        routes = route_lookup.get_group(key)
        snap_frame = first_event_frame(play_tracking, {"ball_snap"})
        play_row=play_lookup.loc[key]
        context = (float(play_row.get("down", 0.0)), float(play_row.get("yardsToGo", 0.0)), float(play_row.get("absoluteYardlineNumber", 0.0)), float((pass_frame - snap_frame) / FRAMES_PER_SECOND) if snap_frame is not None else 0.0)
        for row in routes.itertuples(index=False):
            yac = max(float(row.yardageGainedAfterTheCatch), 0.0)
            receiving_yards = float(row.receivingYards)
            display_name = display_lookup.get((game_id, play_id, row.nflId), "")
            opportunities.append(OpportunityIndex(game_id=int(game_id), play_id=int(play_id), target_nfl_id=int(row.nflId), pass_frame=int(pass_frame), display_name=str(display_name), context=context, was_actual_target=float(row.wasTargettedReceiver), completion=float(row.hadPassReception), interception=float(play_lookup.loc[key, "passResult"] == "IN"), air_yards=receiving_yards - yac,yac=yac))
    return tracking, opportunities


def fit_scalers(tracking: pd.DataFrame, samples: list[SampleIndex], max_players: int) -> tuple[StandardScaler, StandardScaler]:
    rows =[]
    sample_lookup ={(sample.game_id, sample.play_id): sample for sample in samples[: min(len(samples), 1000)]}
    for key, play in tracking.groupby(["gameId", "playId"], sort=False):
        sample = sample_lookup.get(key)
        if sample is None:
            continue
        frame = play.loc[play["frameId"].eq(sample.pass_frame)].copy()
        if frame.empty:
            continue
        target = frame.loc[frame["nflId"].eq(sample.target_nfl_id)]
        qb = frame.loc[frame["is_qb"].eq(1)]
        target_x = target["x_norm"].iloc[0] if not target.empty else frame["x_norm"].mean()
        target_y = target["y_norm"].iloc[0] if not target.empty else frame["y_norm"].mean()
        qb_x = qb["x_norm"].iloc[0] if not qb.empty else frame["x_norm"].mean()
        qb_y = qb["y_norm"].iloc[0] if not qb.empty else frame["y_norm"].mean()
        frame["is_target"] = frame["nflId"].eq(sample.target_nfl_id).astype(float)
        frame["distance_to_qb"] = np.hypot(frame["x_norm"] - qb_x, frame["y_norm"] - qb_y)
        frame["distance_to_target"] = np.hypot(frame["x_norm"] - target_x, frame["y_norm"] - target_y)
        frame["x_from_qb"] =frame["x_norm"]-qb_x
        frame["y_from_qb"] =frame["y_norm"]-qb_y
        frame["x_from_target"] =frame["x_norm"]-target_x
        frame["y_from_target"] =frame["y_norm"]-target_y
        frame = add_graph_context_features(frame, target_x, target_y, qb_x, qb_y, qb)
        rows.append(prioritize_players(frame, sample.target_nfl_id, max_players)[NODE_FEATURES])
    node_scaler = StandardScaler()
    if rows:
        node_scaler.fit(pd.concat(rows).replace([np.inf, -np.inf],np.nan).fillna(0.0))
    else:
        node_scaler.fit(np.zeros((1,len(NODE_FEATURES))))

    context_scaler = StandardScaler()
    if samples:
        context_scaler.fit(np.array([sample.context for sample in samples], dtype=np.float32))
    else:
        context_scaler.fit(np.zeros((1,len(CONTEXT_FEATURES))))
    return node_scaler,context_scaler


def train_one_epoch( model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str, yard_loss_weight: float, air_yards_loss_weight: float, yac_loss_weight: float, interception_loss_weight: float, interception_pos_weight: float, prediction_task: str) -> float:
    model.train()
    total_loss = 0.0
    bce = nn.BCEWithLogitsLoss()
    interception_bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(interception_pos_weight), device=device))
    mae = nn.SmoothL1Loss()
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        out = model(batch["nodes"], batch["context"], batch["mask"], batch["target_index"])
        actual_yards = batch["air_yards"] + batch["yac"]
        losses = []
        if prediction_task in {"all", "outcome"}:
            losses.append(bce(out["completion_logit"], batch["completion"]))
            losses.append(interception_loss_weight * interception_bce(out["interception_logit"], batch["interception"]))
        if prediction_task in {"all", "total_yards"}:
            weight = yard_loss_weight if prediction_task == "all" else 1.0
            losses.append(weight * mae(out["total_yards"], actual_yards))
        if prediction_task in {"all", "air_yards"}:
            weight = air_yards_loss_weight if prediction_task == "all" else 1.0
            losses.append(weight * mae(out["air_yards"], batch["air_yards"]))
        if prediction_task in {"all", "yac"}:
            weight = yac_loss_weight if prediction_task == "all" else 1.0
            losses.append(weight * mae(out["yac"], batch["yac"]))
        loss = sum(losses)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(batch["completion"])
    return total_loss / max(len(loader.dataset), 1)


def best_metric_name(prediction_task: str) -> str:
    if prediction_task == "outcome":
        return "outcome_brier"
    if prediction_task == "air_yards":
        return "air_yards_mae"
    if prediction_task == "yac":
        return "yac_mae"
    return "yards_mae"


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float | None]:
    model.eval()
    rows = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(batch["nodes"], batch["context"], batch["mask"], batch["target_index"])
        rows.append(pd.DataFrame({ "completion": batch["completion"].cpu().numpy(), "completion_probability": torch.sigmoid(out["completion_logit"]).cpu().numpy(), "interception": batch["interception"].cpu().numpy(), "interception_probability": torch.sigmoid(out["interception_logit"]).cpu().numpy(), "actual_air_yards": batch["air_yards"].cpu().numpy(), "predicted_air_yards": out["air_yards"].cpu().numpy(), "actual_yac": batch["yac"].cpu().numpy(), "predicted_yac": out["yac"].cpu().numpy(), "actual_yards": (batch["air_yards"] + batch["yac"]).cpu().numpy(), "predicted_yards": out["total_yards"].cpu().numpy(), "predicted_yards_from_parts": (out["air_yards"] + out["yac"]).cpu().numpy()}))
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    metrics: dict[str, float | None] = {
        "yards_mae": float(mean_absolute_error(df["actual_yards"], df["predicted_yards"])),
        "air_yards_mae": float(mean_absolute_error(df["actual_air_yards"], df["predicted_air_yards"])),"yac_mae": float(mean_absolute_error(df["actual_yac"], df["predicted_yac"])), "yards_from_parts_mae": float( mean_absolute_error(df["actual_yards"], df["predicted_yards_from_parts"])),
        "completion_brier": float(brier_score_loss(df["completion"], df["completion_probability"])),
        "interception_brier": float(brier_score_loss(df["interception"], df["interception_probability"])),
        "outcome_brier": float((brier_score_loss(df["completion"], df["completion_probability"]) + brier_score_loss(df["interception"], df["interception_probability"])) / 2.0),
        "completion_auc": None,
        "interception_auc": None,
    }
    if df["completion"].nunique() > 1:
        metrics["completion_auc"] = float(roc_auc_score(df["completion"], df["completion_probability"]))
    if df["interception"].nunique() > 1:
        metrics["interception_auc"] = float(
            roc_auc_score(df["interception"], df["interception_probability"])
        )
    return metrics


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    opportunities: list[OpportunityIndex],
    device: str,
    output_path: Path,
) -> None:
    model.eval()
    rows = []
    offset = 0
    for batch in loader:
        batch_size = len(batch["completion"])
        batch_on_device = {key: value.to(device) for key, value in batch.items()}
        out = model(
            batch_on_device["nodes"],
            batch_on_device["context"],
            batch_on_device["mask"],
            batch_on_device["target_index"],
        )
        completion_probability = torch.sigmoid(out["completion_logit"]).cpu().numpy()
        interception_probability = torch.sigmoid(out["interception_logit"]).cpu().numpy()
        air_yards = out["air_yards"].cpu().numpy()
        yac = out["yac"].cpu().numpy()
        total_yards = out["total_yards"].cpu().numpy()
        batch_opportunities = opportunities[offset : offset + batch_size]
        offset += batch_size
        for i, opportunity in enumerate(batch_opportunities):
            predicted_yards = float(total_yards[i])
            risk_adjusted_yards = float(
                completion_probability[i] * predicted_yards
                + interception_probability[i] * -45.0
            )
            rows.append({ "gameId": opportunity.game_id,"playId": opportunity.play_id, "frameId": opportunity.pass_frame,
                    "nflId": opportunity.target_nfl_id,
                    "displayName": opportunity.display_name,
                    "was_actual_target": opportunity.was_actual_target,
                    "actual_completion": opportunity.completion,
                    "actual_interception": opportunity.interception,
                    "actual_receiving_yards": opportunity.air_yards + opportunity.yac,
                    "graph_completion_probability": float(completion_probability[i]),
                    "graph_interception_probability": float(interception_probability[i]),
                    "graph_predicted_air_yards_if_completed": float(air_yards[i]),
                    "graph_predicted_yac_if_completed": float(yac[i]),
                    "graph_predicted_yards_from_parts": float(air_yards[i] + yac[i]),
                    "graph_predicted_yards_if_completed": predicted_yards,
                    "graph_risk_adjusted_expected_yards": risk_adjusted_yards,
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def load_graph_model_from_metadata(
    checkpoint_path: Path,
    metadata_path: Path,
    device: str,
) -> tuple[nn.Module, dict]:
    metadata = json.loads(metadata_path.read_text())
    model = make_graph_model(
        architecture=metadata.get("architecture", "transformer"),
        node_feature_dim=len(metadata["node_features"]),
        context_feature_dim=len(metadata.get("context_features", CONTEXT_FEATURES)),
        hidden_dim=int(metadata["hidden_dim"]),
        num_heads=int(metadata["num_heads"]),
        num_layers=int(metadata["num_layers"]),
        max_players=int(metadata["max_players"]),
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, metadata


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tracking = load_tracking(args.data_dir, args.weeks)
    plays = read_csv(args.data_dir, "plays.csv")
    player_play = read_csv(args.data_dir, "player_play.csv")
    players = read_csv(args.data_dir, "players.csv")

    if args.mode == "infer":
        checkpoint = args.checkpoint or args.output_dir / "spatiotemporal_graph_transformer.pt"
        metadata_path = args.metadata or args.output_dir / "metadata.json"
        scaler_path = args.output_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Could not find saved scaler at {scaler_path}")
        with scaler_path.open("rb") as f:
            scalers = pickle.load(f)
        if isinstance(scalers, dict):
            node_scaler = scalers["node_scaler"]
            context_scaler = scalers["context_scaler"]
        else:
            node_scaler = scalers
            context_scaler = StandardScaler().fit(np.zeros((1, len(CONTEXT_FEATURES))))
        model, metadata = load_graph_model_from_metadata(checkpoint, metadata_path, args.device)
        tracking, opportunities = build_inference_opportunities( tracking=tracking, plays=plays, player_play=player_play, players=players, max_plays=args.max_plays)
        dataset = PassingGraphDataset(tracking=tracking, samples=opportunities, node_scaler=node_scaler, context_scaler=context_scaler, sequence_length=int(metadata["sequence_length"]), max_players=int(metadata["max_players"]))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        output_path = args.inference_output or args.output_dir / "graph_receiver_opportunities.csv"
        run_inference(model, loader, opportunities, args.device, output_path)
        print(f"Wrote {len(opportunities):,} graph receiver-opportunity rows to {output_path}")
        return

    tracking, samples = build_samples(
        tracking=tracking,
        plays=plays,
        player_play=player_play,
        players=players,
        max_plays=args.max_plays,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    node_scaler, context_scaler = fit_scalers(tracking, train_samples, args.max_players)

    train_dataset = PassingGraphDataset(
        tracking, train_samples, node_scaler, context_scaler, args.sequence_length, args.max_players
    )
    test_dataset = PassingGraphDataset(
        tracking, test_samples, node_scaler, context_scaler, args.sequence_length, args.max_players
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = make_graph_model(
        architecture=args.architecture,
        node_feature_dim=len(NODE_FEATURES),
        context_feature_dim=len(CONTEXT_FEATURES),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_players=args.max_players,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_interceptions = sum(sample.interception for sample in train_samples)
    train_non_interceptions = len(train_samples) - train_interceptions
    interception_pos_weight = 1.0
    if train_interceptions > 0 and train_non_interceptions > 0:
        interception_pos_weight = min(float(args.interception_pos_weight_cap), float(train_non_interceptions / train_interceptions))

    history = []
    best_metric = float("inf")
    best_epoch = 0
    no_improve = 0
    best_path = args.output_dir / "spatiotemporal_graph_transformer.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            args.device,
            yard_loss_weight=args.yard_loss_weight,
            air_yards_loss_weight=args.air_yards_loss_weight,
            yac_loss_weight=args.yac_loss_weight,
            interception_loss_weight=args.interception_loss_weight,
            interception_pos_weight=interception_pos_weight,
            prediction_task=args.prediction_task,
        )
        metrics = evaluate(model, test_loader, args.device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)
        print(f"epoch={epoch} train_loss={train_loss:.4f} metrics={metrics}")
        metric_name = best_metric_name(args.prediction_task)
        current_metric = float(metrics.get(metric_name, float("inf")))
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

    with (args.output_dir / "feature_scaler.pkl").open("wb") as f:
        pickle.dump({"node_scaler": node_scaler, "context_scaler": context_scaler}, f)
    metric_name = best_metric_name(args.prediction_task)
    metadata = { "architecture": args.architecture, "prediction_task": args.prediction_task, "weeks": args.weeks, "train_samples": len(train_samples), "test_samples": len(test_samples), "sequence_length": args.sequence_length, "max_players": args.max_players, "hidden_dim": args.hidden_dim, "num_heads": args.num_heads, "num_layers": args.num_layers, "node_features": NODE_FEATURES, "context_features": CONTEXT_FEATURES, "best_epoch": best_epoch, "best_metric_name": metric_name, "best_metric": best_metric, "best_yards_mae": best_metric if metric_name == "yards_mae" else None, "yard_loss_weight": args.yard_loss_weight, "air_yards_loss_weight": args.air_yards_loss_weight, "yac_loss_weight": args.yac_loss_weight, "interception_loss_weight": args.interception_loss_weight, "interception_pos_weight": interception_pos_weight, "interception_pos_weight_cap": args.interception_pos_weight_cap, "history": history }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
