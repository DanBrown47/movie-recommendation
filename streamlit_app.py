#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yaml

# Local modules (from the earlier setup)
from data_io import load_items_csv, load_ratings_csv, compute_popular_ids
from recommender_core import (
    build_item_texts, embed_texts, l2norm, FaissStore,
    recommend_for_user,
)

# ---------- Config ----------
CFG_PATH = "config.yaml"
cfg = yaml.safe_load(open(CFG_PATH))
DATA_DIR = Path("./data")
ART_DIR = Path(cfg["artifacts"]["dir"])
ART_DIR.mkdir(parents=True, exist_ok=True)

ITEM_EMBS_NPY = ART_DIR / cfg["artifacts"]["item_embs"]
ITEM_IDS_NPY  = ART_DIR / cfg["artifacts"]["item_ids"]
POPULAR_NPY   = ART_DIR / cfg["artifacts"]["popular_ids"]
FAISS_PATH    = ART_DIR / cfg["artifacts"]["faiss_index"]

# ---------- Ingest ML-100k -> CSV ----------
GENRES_100K = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

def ingest_ml100k_to_csv(src_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # u.item
    cols = ["item_id","title","release_date","video_release_date","imdb_url"] + GENRES_100K
    items = pd.read_csv(src_dir/"u.item", sep="|", header=None, names=cols, encoding="latin-1", engine="python")
    for g in GENRES_100K:
        items[g] = items[g].fillna(0).astype(int)
    def row_to_genres(r):
        tags = [g for g in GENRES_100K if int(r[g]) == 1]
        if "unknown" in tags and len(tags) > 1:
            tags = [g for g in tags if g != "unknown"]
        return "|".join(tags)
    items["genres"] = items.apply(row_to_genres, axis=1)
    items = items[["item_id","title","genres"]]

    # u.data
    ratings = pd.read_csv(src_dir/"u.data", sep="\t", header=None,
                          names=["user_id","item_id","rating","timestamp"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)

    items.to_csv(out_dir/"items.csv", index=False)
    ratings.to_csv(out_dir/"ratings.csv", index=False)

# ---------- Caches ----------
@st.cache_data(show_spinner=False)
def cached_load_data(ratings_csv: str, items_csv: str):
    ratings = load_ratings_csv(ratings_csv)
    items   = load_items_csv(items_csv)
    return ratings, items

@st.cache_resource(show_spinner=False)
def cached_load_faiss(index_path: str):
    return FaissStore.load(index_path)

@st.cache_resource(show_spinner=True)
def cached_build_artifacts(ratings_csv: str, items_csv: str, model_id: str, batch_size: int):
    ratings, items = cached_load_data(ratings_csv, items_csv)
    texts = build_item_texts(items)
    embs = embed_texts(model_id, texts, batch_size=batch_size)
    embs = l2norm(embs)

    np.save(ITEM_EMBS_NPY, embs)
    np.save(ITEM_IDS_NPY, items["item_id"].values)
    pop = compute_popular_ids(ratings, min_positive=float(cfg["recs"]["min_positive"]), limit=5000)
    np.save(POPULAR_NPY, pop)

    store = FaissStore(dim=embs.shape[1])
    store.add(embs)
    store.save(str(FAISS_PATH))
    return True

# ---------- UI ----------
st.set_page_config(page_title="Recsys (Transformers + FAISS)", layout="wide")
st.title("🎬 ML-100k Recommender — Transformers + FAISS + Streamlit")

with st.sidebar:
    st.header("1) Ingest ML-100k → CSV")
    ml_dir = st.text_input("Path to ML-100k folder", value="./ml-100k")
    if st.button("Ingest", use_container_width=True):
        try:
            ingest_ml100k_to_csv(Path(ml_dir), DATA_DIR)
            st.success("Ingest complete → ./data/items.csv & ./data/ratings.csv")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Ingest failed: {e}")

    st.divider()
    st.header("2) Build / Refresh Artifacts")
    model_id = st.text_input("SentenceTransformer model", value=cfg["embeddings"]["model_id"])
    batch_sz = st.number_input("Batch size", min_value=16, max_value=2048, value=int(cfg["embeddings"]["batch_size"]), step=16)
    build_clicked = st.button("Build / Refresh FAISS", type="primary", use_container_width=True)
    if build_clicked:
        try:
            ok = cached_build_artifacts(cfg["data"]["ratings_csv"], cfg["data"]["items_csv"], model_id, batch_sz)
            if ok:
                st.success(f"Artifacts saved → {ART_DIR}")
                st.cache_resource.clear()
        except Exception as e:
            st.error(f"Build failed: {e}")

    st.divider()
    st.header("3) Recommend")
    k = st.number_input("Top-K", min_value=1, max_value=50, value=int(cfg["recs"]["default_k"]))
    fetch = st.number_input("Initial fetch (ANN)", min_value=10, max_value=2000, value=int(cfg["recs"]["fetch"]), step=10)
    mmr_lambda = st.slider("MMR λ (diversity)", 0.0, 1.0, float(cfg["recs"]["mmr_lambda"] or 0.0), help="0 = only diversity, 1 = only relevance. Set to 0.2–0.4 for sane mix.")

st.subheader("Dataset status")
items_csv = cfg["data"]["items_csv"]; ratings_csv = cfg["data"]["ratings_csv"]
data_ok = (Path(items_csv).exists() and Path(ratings_csv).exists())
art_ok  = (ITEM_EMBS_NPY.exists() and ITEM_IDS_NPY.exists() and FAISS_PATH.exists() and POPULAR_NPY.exists())

col1, col2 = st.columns(2)
with col1:
    if data_ok:
        ratings, items = cached_load_data(ratings_csv, items_csv)
        st.write(f"**items.csv**: {len(items)} items")
        st.write(f"**ratings.csv**: {len(ratings):,} ratings by {ratings['user_id'].nunique()} users")
    else:
        st.warning("Data CSVs not found. Ingest ML-100k first.")

with col2:
    if art_ok:
        st.success("Artifacts found ✅ (FAISS + embeddings + popular)")
    else:
        st.warning("Artifacts missing. Build/Refresh FAISS.")

st.divider()

# Recommend UI
if data_ok:
    user_ids = np.sort(ratings["user_id"].unique())
    pick_user = st.selectbox("Pick a user", options=user_ids.tolist(), index=0)

    if st.button("Recommend", type="primary"):
        try:
            if not art_ok:
                st.error("Artifacts missing. Build them in the sidebar first.")
                st.stop()

            item_embs = np.load(ITEM_EMBS_NPY)
            item_ids  = np.load(ITEM_IDS_NPY)
            popular   = np.load(POPULAR_NPY)
            store     = cached_load_faiss(str(FAISS_PATH))

            df = recommend_for_user(
                user_id=int(pick_user),
                ratings=ratings,
                items=items,
                item_embs=item_embs,
                store=store,
                item_ids=item_ids,
                k=int(k),
                fetch=int(fetch),
                min_positive=float(cfg["recs"]["min_positive"]),
                mmr_lambda=float(mmr_lambda),
                popular_ids=popular
            )
            st.subheader(f"Top {k} for user {pick_user}")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Recommendation failed: {e}")
