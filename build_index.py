import yaml, numpy as np
from pathlib import Path
from data_io import load_items_csv, load_ratings_csv, compute_popular_ids
from recommender_core import build_item_texts, embed_texts, l2norm, FaissStore

def main():
    cfg = yaml.safe_load(open("config.yaml"))
    ratings = load_ratings_csv(cfg["data"]["ratings_csv"])
    items   = load_items_csv(cfg["data"]["items_csv"])

    texts = build_item_texts(items)
    embs  = embed_texts(cfg["embeddings"]["model_id"], texts, batch_size=cfg["embeddings"]["batch_size"])
    embs  = l2norm(embs)

    art = Path(cfg["artifacts"]["dir"]); art.mkdir(parents=True, exist_ok=True)
    np.save(art / cfg["artifacts"]["item_embs"], embs)
    np.save(art / cfg["artifacts"]["item_ids"], items["item_id"].values)

    # Precompute popular
    pop = compute_popular_ids(ratings, min_positive=float(cfg["recs"]["min_positive"]), limit=5000)
    np.save(art / cfg["artifacts"]["popular_ids"], pop)

    # Build FAISS
    store = FaissStore(dim=embs.shape[1])
    store.add(embs)
    store.save(str(art / cfg["artifacts"]["faiss_index"]))

    print(f"Saved: {art}")

if __name__ == "__main__":
    main()
