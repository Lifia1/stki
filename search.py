"""TF-IDF (cosine similarity) dan BM25 Okapi untuk pencarian film."""
from __future__ import annotations
import json, math, re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

DATA_CSV_PATH = Path(__file__).parent / "data" / "dataset_preprocessed.csv"
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
        self.films = films
        self.docs = [tokenize(f.get("teks_preprocessed") or f.get("teks_dokumen") or "")
                     for f in films]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        self.tf_docs: List[Counter] = [Counter(d) for d in self.docs]
        self.df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        # precompute tf-idf norms
        self._norms = []
        for tf in self.tf_docs:
            s = 0.0
            for t, c in tf.items():
                w = c * self._idf(t)
                s += w * w
            self._norms.append(math.sqrt(s) or 1.0)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.N + 1) / (n + 1)) + 1.0

    # ---------- TF-IDF cosine ----------
    def search_tfidf(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_tf = Counter(q_tokens)
        q_weights = {t: c * self._idf(t) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
        scores = []
        for i, tf in enumerate(self.tf_docs):
            dot = 0.0
            for t, qw in q_weights.items():
                c = tf.get(t)
                if c:
                    dot += qw * (c * self._idf(t))
            if dot > 0:
                scores.append((i, dot / (q_norm * self._norms[i])))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.films[i], s) for i, s in scores[:top_k]]

    # ---------- BM25 Okapi ----------
    def search_bm25(self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = []
        for i, tf in enumerate(self.tf_docs):
            dl = len(self.docs[i]) or 1
            score = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if not f:
                    continue
                n = self.df.get(t, 0)
                idf_bm = math.log(1 + (self.N - n + 0.5) / (n + 0.5))
                score += idf_bm * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / self.avgdl)))
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self.films[i], s) for i, s in scores[:top_k]]

    def search(self, query: str, model: str = "tfidf", top_k: int = 10):
        return self.search_bm25(query, top_k) if model == "bm25" else self.search_tfidf(query, top_k)

    def similar(self, film: Dict, n: int = 6):
        q = f"{film.get('judul','')} {film.get('genre','')} {(film.get('sinopsis','') or '')[:150]}"
        hits = self.search_tfidf(q, n + 1)
        return [(f, s) for f, s in hits if f["id"] != film["id"]][:n]


def load_films_csv(path: Path) -> List[Dict]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.where(pd.notnull(df), None)
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
    if path.suffix.lower() == ".csv":
        return load_films_csv(path)
    return load_films_json(path)


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
