"""TF-IDF + Cosine Similarity untuk pencarian film.

Mengikuti rumus pada jurnal referensi:
  - TF  : 1 + log10(tf)  jika tf > 0
  - IDF : log10(N / df)
  - Bobot TF-IDF dinormalisasi dengan L2-norm (panjang dokumen)
  - Kemiripan : cosine similarity antar vektor query dan dokumen
"""
from __future__ import annotations
import json, math, re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

DATA_CSV_PATH  = Path(__file__).parent / "data" / "dataset_preprocessed.csv"
DATA_JSON_PATH = Path(__file__).parent / "data" / "films.json"

STOPWORDS = set(
    "yang di ke dari dan atau ini itu pada untuk dengan sebagai adalah akan juga "
    "karena oleh dalam para suatu sebuah the a an of and or to in on for with by film"
    .split()
)


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1 and t not in STOPWORDS]


class SearchEngine:
    def __init__(self, films: List[Dict]):
        self.films   = films
        self.docs    = [
            tokenize(f.get("teks_preprocessed") or f.get("teks_dokumen") or "")
            for f in films
        ]
        self.N       = len(self.docs)
        self.tf_docs : List[Counter] = [Counter(d) for d in self.docs]

        # document frequency: berapa dokumen yang mengandung setiap term
        self.df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1

        # pre-hitung L2-norm tiap dokumen (sesuai Persamaan 4 jurnal)
        self._norms: List[float] = []
        for tf in self.tf_docs:
            s = sum(self._wtf(c) ** 2 * self._idf(t) ** 2
                    for t, c in tf.items())
            self._norms.append(math.sqrt(s) or 1.0)

    # ── rumus jurnal ──────────────────────────────────────────────────────────

    def _wtf(self, tf: int) -> float:
        """TF berbobot log: 1 + log10(tf).  Persamaan 1 jurnal."""
        return (1 + math.log10(tf)) if tf > 0 else 0.0

    def _idf(self, term: str) -> float:
        """IDF klasik: log10(N / df).  Persamaan 2 jurnal."""
        n = self.df.get(term, 0)
        return math.log10(self.N / n) if n > 0 else 0.0

    # ── pencarian utama ───────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """TF-IDF + Cosine Similarity.  Persamaan 3-6 jurnal."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        q_tf      = Counter(q_tokens)
        q_weights = {t: self._wtf(c) * self._idf(t) for t, c in q_tf.items()}
        q_norm    = math.sqrt(sum(w ** 2 for w in q_weights.values())) or 1.0

        scores = []
        for i, tf in enumerate(self.tf_docs):
            dot = sum(
                q_weights[t] * (self._wtf(tf.get(t, 0)) * self._idf(t))
                for t in q_weights
                if tf.get(t, 0)
            )
            if dot > 0:
                # normalisasi — Persamaan 6 (versi dinormalisasi)
                scores.append((i, dot / (q_norm * self._norms[i])))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.films[i], s) for i, s in scores[:top_k]]

    def similar(self, film: Dict, n: int = 6) -> List[Tuple[Dict, float]]:
        judul    = film.get('judul', '')
        genre    = film.get('genre', '')
        sinopsis = (film.get('sinopsis', '') or '')[:300]   
        pemain   = (film.get('pemain', '') or '')[:100]
        sutradara = film.get('sutradara', '') or ''
        
        q = f"{judul} {judul} {genre} {genre} {sinopsis} {pemain} {sutradara}"
        hits = self.search(q, top_k=n + 10)
        return [(f, s) for f, s in hits if f["id"] != film["id"]][:n]
# ── loader ────────────────────────────────────────────────────────────────────

def load_films_csv(path: Path) -> List[Dict]:
    df      = pd.read_csv(path, encoding="utf-8-sig")
    df      = df.where(pd.notnull(df), None)
    records = df.to_dict(orient="records")
    for r in records:
        if isinstance(r.get("id"), float) and r["id"] == int(r["id"]):
            r["id"] = int(r["id"])
    return records


def load_films(path: Path | None = None) -> List[Dict]:
    if path is None:
        if DATA_CSV_PATH.exists():
            return load_films_csv(DATA_CSV_PATH)
        return load_films_json(DATA_JSON_PATH)
    return load_films_csv(path) if path.suffix.lower() == ".csv" else load_films_json(path)


def load_films_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def all_genres(films: List[Dict]) -> List[str]:
    s = set()
    for f in films:
        for g in (f.get("genre") or "").split(","):
            g = g.strip()
            if g:
                s.add(g)
    return sorted(s)