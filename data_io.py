import pandas as pd
from pathlib import Path
from typing import Tuple

def load_ratings_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    req = {"user_id", "item_id", "rating", "timestamp"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"ratings.csv missing columns: {missing}")
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df

def load_items_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p)
    req = {"item_id", "title", "genres"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"items.csv missing columns: {missing}")
    df["item_id"] = df["item_id"].astype(int)
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)
    return df

def compute_popular_ids(ratings: pd.DataFrame, min_positive: float = 4.0, limit: int = 1000):
    pop = (
        ratings[ratings["rating"] >= min_positive]
        .groupby("item_id").size().sort_values(ascending=False).index.values
    )
    return pop[:limit]
