import yaml, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from pathlib import Path
from data_io import load_items_csv, load_ratings_csv
from recommender_core import FaissStore, recommend_for_user

app = FastAPI(title="Recsys Service", version="1.0")

class RecResponse(BaseModel):
    user_id: int
    k: int
    items: list

# Load once at startup
cfg = yaml.safe_load(open("config.yaml"))
ART = Path(cfg["artifacts"]["dir"])
items = load_items_csv(cfg["data"]["items_csv"])
ratings = load_ratings_csv(cfg["data"]["ratings_csv"])

item_embs = np.load(ART / cfg["artifacts"]["item_embs"])
item_ids  = np.load(ART / cfg["artifacts"]["item_ids"])
popular   = np.load(ART / cfg["artifacts"]["popular_ids"])

store = FaissStore.load(str(ART / cfg["artifacts"]["faiss_index"]))

@app.get("/healthz")
def healthz():
    return {"ok": True, "items": int(len(items)), "ratings": int(len(ratings))}

@app.get("/recommend", response_model=RecResponse)
def recommend(user_id: int = Query(..., ge=1), k: int = Query(None, ge=1), mmr: float | None = Query(None), fetch: int | None = Query(None)):
    try:
        k = k or int(cfg["recs"]["default_k"])
        fetch = fetch or int(cfg["recs"]["fetch"])
        mmr_lambda = mmr if mmr is not None else cfg["recs"]["mmr_lambda"]
        min_positive = float(cfg["recs"]["min_positive"])

        df = recommend_for_user(
            user_id=user_id,
            ratings=ratings,
            items=items,
            item_embs=item_embs,
            store=store,
            item_ids=item_ids,
            k=k,
            fetch=fetch,
            min_positive=min_positive,
            mmr_lambda=mmr_lambda,
            popular_ids=popular
        )
        return {
            "user_id": user_id,
            "k": k,
            "items": df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
