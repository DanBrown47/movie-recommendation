#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import yaml
import json

# Local modules (from the earlier setup)
from data_io import load_items_csv, load_ratings_csv, compute_popular_ids
from recommender_core import (
    build_item_texts, embed_texts, l2norm, FaissStore,
    recommend_for_user,
)

# testing stuff
import math
from typing import Dict, Tuple, List


CFG_PATH = "config.yaml"
cfg = yaml.safe_load(open(CFG_PATH))
DATA_DIR = Path("./data")
ART_DIR = Path(cfg["artifacts"]["dir"])
ART_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_PATH = ART_DIR / "feedback.json" 
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

# ---- Session state flags to keep the recs panel visible across reruns
if "show_recs" not in st.session_state:
    st.session_state.show_recs = False
if "last_user" not in st.session_state:
    st.session_state.last_user = None
if "last_k" not in st.session_state:
    st.session_state.last_k = 10
if "last_fetch" not in st.session_state:
    st.session_state.last_fetch = 200
if "last_mmr" not in st.session_state:
    st.session_state.last_mmr = 0.2


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


def load_feedback() -> dict:
    if FEEDBACK_PATH.exists():
        try:
            return json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_feedback(fb: dict) -> None:
    FEEDBACK_PATH.write_text(json.dumps(fb, indent=2))

def record_feedback(user_id: int, item_id: int, label: str) -> None:
    fb = load_feedback()
    ukey = str(user_id)
    st.toast(f"Recording {label} for user {user_id}, item {item_id}...")
    u = fb.get(ukey, {"like": [], "dislike": []})
    # remove from the opposite bucket
    if label == "like":
        if item_id not in u["like"]:
            u["like"].append(item_id)
        u["dislike"] = [i for i in u["dislike"] if i != item_id]
    else:
        if item_id not in u["dislike"]:
            u["dislike"].append(item_id)
        u["like"] = [i for i in u["like"] if i != item_id]
    fb[ukey] = u
    save_feedback(fb)
    st.toast(f"Saved {label}: {item_id}", icon="👍" if label == "like" else "👎")

import math
from typing import Dict, Tuple, List

def _dcg_at_k(rank: int) -> float:
    # leave-one-out binary relevance: dcg = 1 / log2(rank+1)
    return 1.0 / math.log2(rank + 1) if rank > 0 else 0.0

def _intra_list_diversity(item_ids: np.ndarray, item_embs_map: Dict[int, int], item_embs: np.ndarray) -> float:
    # 1 - average pairwise cosine among the K recs
    idxs = [item_embs_map.get(i, -1) for i in item_ids]
    idxs = [i for i in idxs if i >= 0]
    if len(idxs) < 2:
        return 0.0
    V = item_embs[idxs]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T  # cosine
    m = S.shape[0]
    off_diag_sum = (S.sum() - np.trace(S))
    pairs = m * (m - 1)
    avg_cos = off_diag_sum / pairs
    return float(1.0 - avg_cos)

def _novelty(item_ids: np.ndarray, pop_counts: Dict[int, int], total_pos: int) -> float:
    # higher is more novel
    vals = []
    for iid in item_ids:
        p = pop_counts.get(int(iid), 0) + 1e-9
        vals.append(-math.log(p / max(total_pos, 1)))
    return float(np.mean(vals)) if vals else 0.0

def _leave_one_out_split(ratings: pd.DataFrame, min_positive: float = 4.0) -> Dict[int, Tuple[pd.DataFrame, Tuple[int,int,int]]]:
    """
    For each user with >=2 positives, hold out the latest positive (by timestamp).
    Returns { user_id: (ratings_train_user_df, (holdout_item_id, holdout_ts, n_train_pos)) }
    """
    out = {}
    pos = ratings[ratings["rating"] >= min_positive].copy()
    if pos.empty:
        return out
    pos = pos.sort_values(["user_id","timestamp"])
    for uid, grp in pos.groupby("user_id"):
        if len(grp) < 2:
            continue
        ho = grp.iloc[-1]  # last positive
        # build a per-user train df: all user rows except the exact holdout row
        # (Use index to avoid collisions)
        user_all = ratings[ratings["user_id"] == uid]
        mask = ~((user_all["item_id"] == ho["item_id"]) & (user_all["timestamp"] == ho["timestamp"]))
        train_user = user_all[mask].copy()
        out[int(uid)] = (train_user, (int(ho["item_id"]), int(ho["timestamp"]), int((grp.shape[0]-1))))
    return out

def run_offline_eval(
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    store,  # FaissStore
    item_ids: np.ndarray,
    k: int,
    fetch: int,
    min_positive: float,
    mmr_lambda: float | None,
    user_sample: int | None = None,
    respect_dislikes: bool = False,
    feedback: dict | None = None,
) -> Dict[str, float]:
    # popularity for novelty
    pos = ratings[ratings["rating"] >= min_positive]
    pop_counts = pos.groupby("item_id").size().to_dict()
    total_pos = int(len(pos))
    # item id -> row index in emb matrix
    emb_row = {int(i): idx for idx, i in enumerate(item_ids)}

    splits = _leave_one_out_split(ratings, min_positive=min_positive)
    users = list(splits.keys())
    if user_sample is not None and user_sample < len(users):
        users = users[:user_sample]  # simple slice; could randomize if you want

    hits = 0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    cov = set()
    divs = []
    novs = []

    for uid in users:
        train_user, (ho_item, _, _) = splits[uid]

        # optional exclusions (dislikes)
        exclude = set()
        if respect_dislikes and feedback:
            exclude = set(feedback.get(str(uid), {}).get("dislike", []))

        # recommend using only the user's TRAIN interactions
        df = recommend_for_user(
            user_id=uid,
            ratings=train_user,     # <- prevent leakage
            items=items,
            item_embs=item_embs,
            store=store,
            item_ids=item_ids,
            k=k,
            fetch=fetch,
            min_positive=min_positive,
            mmr_lambda=mmr_lambda,
            popular_ids=np.array([]),  # unused path here
            exclude_ids=exclude,
        )

        rec_ids = df["item_id"].to_numpy(dtype=int)
        cov.update(rec_ids.tolist())

        # metrics vs holdout
        try:
            rank_pos = int(np.where(rec_ids == ho_item)[0][0]) + 1  # 1-based
            hits += 1
            mrr_sum += 1.0 / rank_pos
            ndcg_sum += _dcg_at_k(rank_pos)
        except IndexError:
            # not found in top-K
            pass
        except Exception:
            pass

        # list-level metrics
        divs.append(_intra_list_diversity(rec_ids, emb_row, item_embs))
        novs.append(_novelty(rec_ids, pop_counts, total_pos))

    n = max(len(users), 1)
    K = max(k, 1)
    # Ideal DCG for binary with single positive is 1/log2(1+1)=1
    return {
        "users_eval": float(len(users)),
        "HitRate@K": hits / n,
        "MRR": mrr_sum / n,
        "NDCG@K": ndcg_sum / n,
        "Diversity": float(np.mean(divs)) if divs else 0.0,
        "Coverage": len(cov) / max(len(items), 1),
        "Novelty": float(np.mean(novs)) if novs else 0.0,
    }


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

    # CHANGED: click sets persistent state, does NOT gate rendering
    if st.button("Recommend", type="primary"):
        st.session_state.show_recs = True
        st.session_state.last_user  = int(pick_user)
        st.session_state.last_k     = int(k)
        st.session_state.last_fetch = int(fetch)
        st.session_state.last_mmr   = float(mmr_lambda)

# CHANGED: render recs whenever show_recs is True
if data_ok and st.session_state.get("show_recs"):
    try:
        if not art_ok:
            st.error("Artifacts missing. Build them in the sidebar first.")
            st.stop()

        # Load artifacts (cached)
        item_embs = np.load(ITEM_EMBS_NPY)
        item_ids  = np.load(ITEM_IDS_NPY)
        popular   = np.load(POPULAR_NPY)
        store     = cached_load_faiss(str(FAISS_PATH))

        # Use the last requested params from session_state
        _user  = int(st.session_state.last_user)
        _k     = int(st.session_state.last_k)
        _fetch = int(st.session_state.last_fetch)
        _mmr   = float(st.session_state.last_mmr)

        fb = load_feedback()
        disliked_set = set(fb.get(str(_user), {}).get("dislike", []))  # exclude these

        df = recommend_for_user(
            user_id=_user,
            ratings=ratings,
            items=items,
            item_embs=item_embs,
            store=store,
            item_ids=item_ids,
            k=_k,
            fetch=_fetch,
            min_positive=float(cfg["recs"]["min_positive"]),
            mmr_lambda=_mmr,
            popular_ids=popular,
            exclude_ids=disliked_set,
        )

        st.subheader(f"Top {_k} for user {_user}")
        for _, row in df.iterrows():
            item_id = int(row["item_id"])
            c1, c2, c3, c4 = st.columns([6, 2, 1, 1])
            with c1:
                st.markdown(f"**{row['title']}**  \n_{row['genres']}_  \n`id={item_id}`")
            with c2:
                try:
                    st.metric("score", f"{float(row.get('score', float('nan'))):.3f}")
                except Exception:
                    st.write("")
            with c3:
                if st.button("👍", key=f"like_{_user}_{item_id}"):
                    record_feedback(_user, item_id, "like")
                    st.rerun()  # NEW: immediately refresh with updated feedback
            with c4:
                if st.button("👎", key=f"dislike_{_user}_{item_id}"):
                    record_feedback(_user, item_id, "dislike")
                    st.rerun()  # NEW

        # Optional: a hide toggle
        if st.button("Hide recommendations"):
            st.session_state.show_recs = False
            st.rerun()

    except Exception as e:
        st.error(f"Recommendation failed: {e}")

st.divider()
st.header("4) Feedback")
if data_ok:
    fb = load_feedback()
    ufb = fb.get(str(pick_user), {"like": [], "dislike": []})
    st.caption("Your stored preferences for the selected user")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**👍 Liked**")
        st.code(", ".join(map(str, ufb.get("like", []))) or "—")
    with c2:
        st.markdown("**👎 Disliked (will be excluded)**")
        st.code(", ".join(map(str, ufb.get("dislike", []))) or "—")
    if st.button("Clear this user's feedback"):
        fb[str(pick_user)] = {"like": [], "dislike": []}
        save_feedback(fb)
        st.success("Cleared.")

st.divider()
st.header("5) Offline Evaluation")

if data_ok and art_ok:
    eval_k = st.number_input("K for metrics", min_value=1, max_value=50, value=int(k))
    eval_fetch = st.number_input("Fetch (ANN) for eval", min_value=10, max_value=2000, value=int(fetch), step=10)
    sample_n = st.number_input("Users to evaluate (leave-one-out)", min_value=10, max_value=943, value=300, step=10)
    respect_dislikes = st.checkbox("Respect current dislikes during eval", value=False)
    compare_mmr = st.checkbox("Compare with MMR off (side-by-side)", value=True)

    if st.button("Run Evaluation", type="secondary"):
        try:
            item_embs = np.load(ITEM_EMBS_NPY)
            item_ids  = np.load(ITEM_IDS_NPY)
            store     = cached_load_faiss(str(FAISS_PATH))
            fb        = load_feedback()

            with st.spinner("Computing metrics..."):
                m1 = run_offline_eval(
                    ratings=ratings,
                    items=items,
                    item_embs=item_embs,
                    store=store,
                    item_ids=item_ids,
                    k=int(eval_k),
                    fetch=int(eval_fetch),
                    min_positive=float(cfg["recs"]["min_positive"]),
                    mmr_lambda=float(mmr_lambda),
                    user_sample=int(sample_n),
                    respect_dislikes=bool(respect_dislikes),
                    feedback=fb,
                )

                if compare_mmr:
                    m0 = run_offline_eval(
                        ratings=ratings,
                        items=items,
                        item_embs=item_embs,
                        store=store,
                        item_ids=item_ids,
                        k=int(eval_k),
                        fetch=int(eval_fetch),
                        min_positive=float(cfg["recs"]["min_positive"]),
                        mmr_lambda=None,  # <- no MMR
                        user_sample=int(sample_n),
                        respect_dislikes=bool(respect_dislikes),
                        feedback=fb,
                    )
                    dfm = pd.DataFrame([{"Setting":"MMR OFF", **m0},{"Setting":"MMR ON", **m1}]).set_index("Setting")
                    st.subheader("Metrics (leave-one-out, last positive per user)")
                    st.dataframe(dfm, use_container_width=True)
                else:
                    st.subheader("Metrics (leave-one-out, last positive per user)")
                    st.dataframe(pd.DataFrame([m1]), use_container_width=True)

                # quick headline metrics
                c1,c2,c3 = st.columns(3)
                c1.metric("HitRate@K", f"{m1['HitRate@K']:.3f}")
                c2.metric("NDCG@K", f"{m1['NDCG@K']:.3f}")
                c3.metric("MRR", f"{m1['MRR']:.3f}")

        except Exception as e:
            st.error(f"Eval failed: {e}")
else:
    st.info("Provide data and build artifacts to run evaluation.")
