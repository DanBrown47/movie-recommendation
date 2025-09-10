#!/usr/bin/env python3
import argparse
import pandas as pd
from pathlib import Path

GENRES_100K = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

def load_u_item(src: Path) -> pd.DataFrame:
    cols = (["item_id","title","release_date","video_release_date","imdb_url"] + GENRES_100K)
    df = pd.read_csv(src/"u.item", sep="|", header=None, names=cols, encoding="latin-1", engine="python")
    for g in GENRES_100K:
        df[g] = df[g].fillna(0).astype(int)
    def row_to_genres(r):
        tags = [g for g in GENRES_100K if int(r[g]) == 1]
        if "unknown" in tags and len(tags) > 1:
            tags = [g for g in tags if g != "unknown"]
        return "|".join(tags)
    df["genres"] = df.apply(row_to_genres, axis=1)
    return df[["item_id","title","genres"]]

def load_u_data(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src/"u.data", sep="\t", header=None,
                     names=["user_id","item_id","rating","timestamp"])
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df

def main():
    ap = argparse.ArgumentParser(description="Convert ML-100k (u.data/u.item) -> ./data/ratings.csv, ./data/items.csv")
    ap.add_argument("--src", required=True, help="Path to ML-100k folder (contains u.data, u.item)")
    ap.add_argument("--out", default="./data", help="Output folder for CSVs")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    items = load_u_item(src)
    ratings = load_u_data(src)

    items.to_csv(out/"items.csv", index=False)
    ratings.to_csv(out/"ratings.csv", index=False)

    print(f"Wrote: {out/'items.csv'} and {out/'ratings.csv'}")

if __name__ == "__main__":
    main()
