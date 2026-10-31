"""News articles: loading, company-name mapping, and causal sentiment aggregation inputs.
Timestamps in the source are day-first 'dd/mm/yyyy hh:mm'; the timezone is NOT documented in the files."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ALIASES = {"egyptianaluminum": "egyptaluminum", "madinetmasr": "madinetnasr"}


def _norm(s: str) -> str:
    s = re.sub(r"^\d+\.\s*", "", s.replace("_", " ")).lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return ALIASES.get(s, s)


def load_news(news_dir, companies) -> pd.DataFrame:
    """Return DataFrame[company, ts, headline, content]; articles de-duplicated; unparseable dates dropped."""
    want = {_norm(c): c for c in companies}
    rows = []
    for f in sorted(Path(news_dir).glob("*.csv")):
        key = _norm(f.stem)
        if key not in want:
            continue
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["ts"] = pd.to_datetime(d["Date_Published"], dayfirst=True, errors="coerce")
        d = d.dropna(subset=["ts"])
        d["company"] = want[key]
        rows.append(d.rename(columns={"Article_Headline": "headline", "Article_Content": "content"})[
            ["company", "ts", "headline", "content"]])
    if not rows:
        return pd.DataFrame(columns=["company", "ts", "headline", "content"])
    n = pd.concat(rows, ignore_index=True)
    return n.drop_duplicates(["company", "ts", "headline"]).sort_values(["company", "ts"]).reset_index(drop=True)
