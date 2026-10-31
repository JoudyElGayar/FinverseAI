"""Score every article once with the pre-trained ProsusAI/finbert model (no training, so no label leakage).
Run on a machine with internet access to download the model (GPU optional; CPU works for a few thousand articles):
    pip install transformers torch
    python scripts/score_news_finbert.py --news data/news --out data/news/sentiment_finbert.csv
Output columns: company, ts, p_pos, p_neg, p_neu, sentiment (= p_pos - p_neg).
NOTE: this script has not been run in the build sandbox (no model download access); test it with --limit 20 first."""
import argparse, sys
import pandas as pd
sys.path.insert(0, "src")
from finverse.data.news import load_news


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--news", default="data/news"); ap.add_argument("--out", default="data/news/sentiment_finbert.csv")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import glob, os
    from finverse.data.news import _norm
    companies = sorted({os.path.basename(f)[:-4] for f in glob.glob(a.news + "/*.csv") if "sentiment" not in f})
    news = load_news(a.news, companies)
    if a.limit: news = news.head(a.limit)
    name = "ProsusAI/finbert"
    tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    texts = (news["headline"].fillna("") + ". " + news["content"].fillna("")).tolist()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), a.batch):
            enc = tok(texts[i:i + a.batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            probs.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            if (i // a.batch) % 20 == 0: print(i, "/", len(texts), flush=True)
    import numpy as np
    P = pd.DataFrame(np.vstack(probs), columns=labels)
    out = pd.concat([news[["company", "ts"]].reset_index(drop=True), P], axis=1)
    out = out.rename(columns={"positive": "p_pos", "negative": "p_neg", "neutral": "p_neu"})
    out["sentiment"] = out["p_pos"] - out["p_neg"]
    out.to_csv(a.out, index=False); print("wrote", a.out, len(out))


if __name__ == "__main__":
    main()
