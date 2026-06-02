"""Compare model errors."""

from __future__ import annotations
import pandas as pd
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a model-error comparison table and graphic.")
    parser.add_argument("--expected-yards-dir",type=Path,default=Path("expected_yards_output_all_weeks"),help="Directory containing tree/XGBoost model_metrics.json.",)
    parser.add_argument("--graph-dir",type=Path,default=Path("graph_transformer_smoke"),help="Directory containing graph transformer metadata.json.",)
    parser.add_argument("--output-dir",type=Path,default=None,help="Directory for comparison outputs. Defaults to expected-yards analysis dir.",)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Could not find{path}")
    return json.loads(path.read_text())


def latest_graph_metrics(metadata: dict) -> dict:
    history = metadata.get("history", [])
    if not history:
        return {}
    best_epoch = metadata.get("best_epoch")
    if best_epoch:
        for row in history:
            if row.get("epoch") == best_epoch:
                return row
    return history[-1]


def build_error_table(tree_dir: Path, graph_dir: Path) -> pd.DataFrame:
    tree = read_json(tree_dir/"model_metrics.json")
    graph = read_json(graph_dir/"metadata.json")
    graph_scores=latest_graph_metrics(graph)

    rows = [
        {"model": tree.get("model_family", "tree_model"),"target": "air_yards","mae": tree.get("air_yards_mae")},
        {"model": tree.get("model_family", "tree_model"),"target": "yards_after_catch","mae": tree.get("yac_mae")},
        {"model": tree.get("model_family", "tree_model"),"target": "total_yards","mae": tree.get("yards_mae")},
        {"model": "gat_temporal_transformer","target": "air_yards","mae": graph_scores.get("air_yards_mae")},
        {"model": "gat_temporal_transformer","target": "yards_after_catch","mae": graph_scores.get("yac_mae")},
        {"model": "gat_temporal_transformer","target": "total_yards","mae": graph_scores.get("yards_mae")},
    ]
    df = pd.DataFrame(rows)
    return df.dropna(subset=["mae"]).reset_index(drop=True)


def write_error_chart(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        raise ValueError("No model error metrics were available to plot.")
    pivot = df.pivot(index="target", columns="model", values="mae")
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_title("Model Error by Prediction Target")
    ax.set_xlabel("Prediction Target")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()



def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.expected_yards_dir/"analysis"
    output_dir.mkdir(parents=True,exist_ok=True)

    df = build_error_table(args.expected_yards_dir, args.graph_dir)
    csv_path = output_dir/"model_error_comparison.csv"
    chart_path = output_dir/"model_error_comparison.png"
    df.to_csv(csv_path,index=False)
    write_error_chart(df,chart_path)
    print(f"Wrote model error table to {csv_path}")
    print(f"Wrote model error graphic to {chart_path}")
if __name__ == "__main__":
    main()
