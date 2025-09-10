import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ---------- Embeddings ----------
def build_item_texts(items: pd.DataFrame) -> List[str]:
    return (items["title"].fillna("") + " | " + items["genres"].fillna("")).str.lower().tolist()

def embed_texts(model_id: str, texts: List[str], batch_size: int = 256) -> np.ndarray:
    model = SentenceTransformer(model_id)
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=False)
    return embs.astype("float32")

def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

# ---------- Vector store ----------
class FaissStore:
    def __init__(self, dim: int):
        if not HAS_FAISS:
            raise RuntimeError("FAISS not available. Install faiss-cpu.")
        self.index = faiss.IndexFlatIP(dim)  # cosine via L2-normalize + inner product

    def add(self, vecs: np.ndarray):
        self.index.add(vecs)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if q.ndim == 1: q = q[None, :]
        sims, idx = self.index.search(q.astype("float32"), k)
        return idx[0], sims[0]

    def save(self, path: str):
        faiss.write_index(self.index, path)

    @staticmethod
    def load(path: str) -> "FaissStore":
        idx = faiss.read_index(path)
        store = FaissStore(idx.d)
        store.index = idx
        return store

# ---------- User profile ----------
def build_user_profile(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    min_positive: float = 4.0,
    tau_days: Optional[float] = 30.0,
) -> Optional[np.ndarray]:
    r = ratings[ratings["user_id"] == user_id]
    pos = r[r["rating"] >= min_positive]
    if pos.empty: return None

    id2row = {iid: i for i, iid in enumerate(items["item_id"].values)}
    rows = np.array([id2row.get(i, -1) for i in pos["item_id"].values])
    mask = rows >= 0
    if not mask.any(): return None

    rows = rows[mask]
    pos = pos.loc[mask]

    w = (pos["rating"].values - 3.0).clip(min=0.5)  # 1..2
    if tau_days and tau_days > 0:
        max_ts = ratings["timestamp"].max()
        age_days = (max_ts - pos["timestamp"].values) / 86400.0
        w = w * np.exp(-age_days / tau_days)

    prof = (item_embs[rows] * w[:, None]).sum(axis=0)
    n = np.linalg.norm(prof) + 1e-12
    return (prof / n).astype("float32")

# ---------- MMR (diversity optional) ----------
def mmr_rerank(cand_vecs: np.ndarray, query_vec: np.ndarray, base_scores: np.ndarray, k: int, lambda_: float = 0.2) -> List[int]:
    selected = []
    candidates = list(range(len(base_scores)))
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    C = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12)
    rel = (C @ q)
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(k, len(base_scores)) and candidates:
        div_terms = C[candidates] @ C[selected].T
        max_div = div_terms.max(axis=1) if div_terms.ndim > 1 else div_terms
        scores = lambda_ * rel[candidates] - (1 - lambda_) * max_div
        j = int(np.argmax(scores))
        selected.append(candidates[j])
        candidates.pop(j)
    return selected

# ---------- Recommendation core ----------
def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    store: FaissStore,
    item_ids: np.ndarray,
    k: int,
    fetch: int,
    min_positive: float,
    mmr_lambda: Optional[float],
    popular_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    prof = build_user_profile(user_id, ratings, items, item_embs, min_positive=min_positive)
    if prof is None:
        ids = popular_ids[:k] if popular_ids is not None else items["item_id"].values[:k]
        return _format(items, ids, None)

    q = prof / (np.linalg.norm(prof) + 1e-12)
    idx, sims = store.search(q, fetch)
    cand_ids = item_ids[idx]
    cand_vecs = item_embs[idx]

    seen = set(ratings.loc[ratings["user_id"] == user_id, "item_id"].values)
    keep = np.array([cid not in seen for cid in cand_ids])
    cand_ids = cand_ids[keep]; cand_vecs = cand_vecs[keep]; sims = sims[keep]

    if cand_ids.size == 0:
        ids = popular_ids[:k] if popular_ids is not None else items["item_id"].values[:k]
        return _format(items, ids, None)

    if mmr_lambda is not None:
        sel = mmr_rerank(cand_vecs, q, sims, k=k, lambda_=mmr_lambda)
        cand_ids, sims = cand_ids[sel], sims[sel]
    else:
        order = np.argsort(-sims)[:k]
        cand_ids, sims = cand_ids[order], sims[order]
    return _format(items, cand_ids, sims)

def _format(items: pd.DataFrame, item_ids: np.ndarray, sims: Optional[np.ndarray]) -> pd.DataFrame:
    it = items.set_index("item_id")
    rows = []
    for r, iid in enumerate(item_ids, start=1):
        title = it.loc[iid, "title"] if iid in it.index else str(iid)
        genres = it.loc[iid, "genres"] if iid in it.index else ""
        score = None if sims is None else float(sims[r-1])
        rows.append({"rank": r, "item_id": int(iid), "title": title, "genres": genres, "score": score})
    return pd.DataFrame(rows)

# recommender_core.py

def recommend_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    item_embs: np.ndarray,
    store: FaissStore,
    item_ids: np.ndarray,
    k: int,
    fetch: int,
    min_positive: float,
    mmr_lambda: Optional[float],
    popular_ids: Optional[np.ndarray] = None,
    exclude_ids: Optional[set[int]] = None,   # 👈 NEW
) -> pd.DataFrame:
    exclude_ids = exclude_ids or set()

    def _fallback(topk: int) -> pd.DataFrame:
        # Popular fallback but respect excludes
        src = popular_ids if popular_ids is not None else items["item_id"].values
        filt = [iid for iid in src if iid not in exclude_ids]
        return _format(items, np.array(filt[:topk]), None)

    prof = build_user_profile(user_id, ratings, items, item_embs, min_positive=min_positive)
    if prof is None:
        return _fallback(k)

    q = prof / (np.linalg.norm(prof) + 1e-12)
    idx, sims = store.search(q, fetch)
    cand_ids = item_ids[idx]
    cand_vecs = item_embs[idx]

    seen = set(ratings.loc[ratings["user_id"] == user_id, "item_id"].values)
    keep = np.array([(cid not in seen) and (cid not in exclude_ids) for cid in cand_ids])
    cand_ids = cand_ids[keep]; cand_vecs = cand_vecs[keep]; sims = sims[keep]

    if cand_ids.size == 0:
        return _fallback(k)

    if mmr_lambda is not None:
        sel = mmr_rerank(cand_vecs, q, sims, k=k, lambda_=mmr_lambda)
        cand_ids, sims = cand_ids[sel], sims[sel]
    else:
        order = np.argsort(-sims)[:k]
        cand_ids, sims = cand_ids[order], sims[order]

    return _format(items, cand_ids, sims)
