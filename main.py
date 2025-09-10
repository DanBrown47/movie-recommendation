#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

# ---- Embeddings ----
from sentence_transformers import SentenceTransformer

# ---- ANN backends ----
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors


# ----------------------------
# Data loading (ML-100k: u.data, u.item)
# ----------------------------
GENRE_COLS = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

def load_ml100k(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["item_id"] = ratings["item_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    ratings["timestamp"] = ratings["timestamp"].astype(int)

    cols = ["item_id","title","release_date","video_release_date","imdb_url"] + GENRE_COLS
    items = pd.read_csv(
        data_dir / "u.item",
        sep="|", header=None, names=cols,
        encoding="latin-1", engine="python"
    )
    for g in GENRE_COLS:
        items[g] = items[g].fillna(0).astype(int)
    def row_to_genres(r):
        tags = [g for g in GENRE_COLS if int(r[g]) == 1]
        if "unknown" in tags and len(tags) > 1:
            tags = [g for g in tags if g != "unknown"]
        return "|".join(tags)
    items["genres"] = items.apply(row_to_genres, axis=1)
    items = items[["item_id","title","genres"]]
    return ratings, items


# ----------------------------
# Vector store abstraction
# ----------------------------
class VectorStore:
    """Cosine similarity ANN over L2-normalized vectors, FAISS if available, else sklearn."""
    def __init__(self, dim: int):
        self.dim = dim
        self._faiss = None
        self._sk = None
        self._vecs = None  # keep normalized for fallback cosine
        self._use_faiss = FAISS_AVAILABLE

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n

    def fit(self, X: np.ndarray):
        X = self._l2norm(X.astype("float32"))
        self._vecs = X
        if self._use_faiss:
            index = faiss.IndexFlatIP(X.shape[1])  # cosine via dot on normalized vectors
            index.add(X)
            self._faiss = index
        else:
            # sklearn cosine distances → convert to similarities in query
            nn = NearestNeighbors(metric="cosine", algorithm="auto")
            nn.fit(X)
            self._sk = nn

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """q: (d,) or (n,d) already L2-normalized or raw; returns (indices, sims)"""
        if q.ndim == 1:
            q = q[None, :]
        q = self._l2norm(q.astype("float32"))

        if self._use_faiss and self._faiss is not None:
            sims, idx = self._faiss.search(q, k)  # sims are dot products in [-1,1]
            return idx[0], sims[0]
        else:
            # sklearn returns distances; similarity = 1 - distance
            dists, idx = self._sk.kneighbors(q, n_neighbors=k, return_distance=True)
            sims = 1.0 - dists
            return idx[0], sims[0]


# ----------------------------
# Embedding utilities
# ----------------------------
def embed_texts(model: SentenceTransformer, texts: List[str], batch_size=256) -> np.ndarray:
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    return embs.astype("float32")

def build_item_texts(items: pd.DataFrame) -> List[str]:
    # Minimal text: title + genres
    return (items["title"].fillna("") + " | " + items["genres"].fillna("")).str.lower().tolist()

def build_user_profile(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    min_positive: float = 4.0,
    tau_days: Optional[float] = 30.0,
) -> Optional[np.ndarray]:
    """Weighted average of embeddings for items the user liked."""
    user_r = ratings[ratings["user_id"] == user_id]
    pos = user_r[user_r["rating"] >= min_positive]
    if pos.empty:
        return None

    # Align to item embedding order
    item_id_to_idx = {iid: i for i, iid in enumerate(items["item_id"].values)}
    rows = [item_id_to_idx.get(i, -1) for i in pos["item_id"].values]
    mask = np.array([r >= 0 for r in rows])
    if not mask.any():
        return None

    rows = np.array(rows)[mask]
    pos = pos.loc[mask]

    # weights: (rating - 3) + time decay
    w = (pos["rating"].values - 3.0).clip(min=0.5)  # 1.0..2.0
    if tau_days and tau_days > 0:
        max_ts = ratings["timestamp"].max()
        age_days = (max_ts - pos["timestamp"].values) / 86400.0
        w = w * np.exp(-age_days / tau_days)

    vecs = item_embs[rows]
    prof = (vecs * w[:, None]).sum(axis=0)
    n = np.linalg.norm(prof) + 1e-12
    return (prof / n).astype("float32")


# ----------------------------
# MMR re-ranking (optional diversity)
# ----------------------------
def mmr_rerank(
    cand_vecs: np.ndarray,
    query_vec: np.ndarray,
    base_scores: np.ndarray,
    k: int,
    lambda_: float = 0.2,
) -> List[int]:
    """Maximal Marginal Relevance: select k indices trading off relevance and diversity."""
    selected = []
    candidates = list(range(len(base_scores)))
    # precompute cosines to query
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    C = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    rel = (C @ q)  # similarity to query
    # start with best relevance
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(k, len(base_scores)) and candidates:
        div_terms = C[candidates] @ C[selected].T  # (C x |S|)
        max_div = div_terms.max(axis=1) if div_terms.ndim > 1 else div_terms
        scores = lambda_ * rel[candidates] - (1 - lambda_) * max_div
        j = int(np.argmax(scores))
        selected.append(candidates[j])
        candidates.pop(j)
    return selected


# ----------------------------
# Recommend for a user
# ----------------------------
def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    store: VectorStore,
    k: int = 10,
    min_positive: float = 4.0,
    mmr_lambda: Optional[float] = None,
    fetch: int = 200
) -> pd.DataFrame:
    prof = build_user_profile(user_id, ratings, items, item_embs, min_positive=min_positive)
    # Popular fallback
    popular = (
        ratings[ratings["rating"] >= min_positive]
        .groupby("item_id").size().sort_values(ascending=False).index.values
    )

    if prof is None:
        top_ids = popular[:k]
        return _format_output(items, top_ids, None)

    # Retrieve ANN neighbors
    idx, sims = store.search(prof, k=fetch)
    item_ids = items["item_id"].values
    cand_ids = item_ids[idx]
    cand_vecs = item_embs[idx]

    # Filter already seen
    seen = set(ratings.loc[ratings["user_id"] == user_id, "item_id"].values.tolist())
    keep_mask = np.array([cid not in seen for cid in cand_ids])
    cand_ids = cand_ids[keep_mask]
    cand_vecs = cand_vecs[keep_mask]
    sims = sims[keep_mask]

    if cand_ids.size == 0:
        top_ids = popular[:k]
        return _format_output(items, top_ids, None)

    # Optional diversity re-rank
    if mmr_lambda is not None and 0.0 <= mmr_lambda <= 1.0:
        sel = mmr_rerank(cand_vecs, prof, sims, k=k, lambda_=mmr_lambda)
        cand_ids = cand_ids[sel]
        sims = sims[sel]
    else:
        # take top-k by similarity
        order = np.argsort(-sims)[:k]
        cand_ids = cand_ids[order]
        sims = sims[order]

    return _format_output(items, cand_ids, sims)


def _format_output(items: pd.DataFrame, item_ids: np.ndarray, sims: Optional[np.ndarray]) -> pd.DataFrame:
    it = items.set_index("item_id")
    rows = []
    for r, iid in enumerate(item_ids, start=1):
        title = it.loc[iid, "title"] if iid in it.index else str(iid)
        genres = it.loc[iid, "genres"] if iid in it.index else ""
        score = None if sims is None else float(sims[r-1])
        rows.append({"rank": r, "item_id": int(iid), "title": title, "genres": genres, "score": score})
    return pd.DataFrame(rows)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Transformer embeddings + vector store (FAISS/Sklearn) for ML-100k.")
    ap.add_argument("--data", required=True, help="Path to ML-100k folder (contains u.data, u.item)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                    help="SentenceTransformers model id")
    ap.add_argument("--user", type=int, default=1, help="User ID to recommend for (1..943)")
    ap.add_argument("--k", type=int, default=10, help="Top-K recommendations")
    ap.add_argument("--fetch", type=int, default=200, help="Initial ANN fetch size before filtering/MMR")
    ap.add_argument("--mmr", type=float, default=None, help="MMR lambda (0..1). Omit for no diversity re-rank")
    ap.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = ap.parse_args()

    data_dir = Path(args.data)
    ratings, items = load_ml100k(data_dir)

    # Build item texts & embeddings
    texts = build_item_texts(items)
    print(f"[emb] loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"[emb] encoding {len(texts)} items...")
    item_embs = embed_texts(model, texts, batch_size=args.batch_size)

    # Build vector store
    store = VectorStore(dim=item_embs.shape[1])
    store.fit(item_embs)

    # Recommend
    df = recommend_for_user(
        user_id=args.user,
        ratings=ratings,
        items=items,
        item_embs=item_embs,
        store=store,
        k=args.k,
        min_positive=4.0,
        mmr_lambda=args.mmr,
        fetch=args.fetch
    )
    pd.set_option("display.max_colwidth", 120)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
