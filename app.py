import os
import sys
import json
import urllib.request
import urllib.error
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from flask import Flask, render_template, request, jsonify
from search import SearchEngine, load_films, all_genres

app = Flask(__name__)

# ============================================================
# TMDB CONFIG
# ============================================================
TMDB_API_KEY    = "222d708c3712f86a090bc1143ef6ae81"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_BASE   = "https://api.themoviedb.org/3/movie"
PLACEHOLDER_IMG = "https://placehold.co/260x390/0f2d1c/839958?text=No+Poster"

# ============================================================
# LOAD DATA & MODEL
# ============================================================
print("⏳ Memuat data dan model...")
FILMS       = load_films()
FILMS_BY_ID = {f["id"]: f for f in FILMS}
ENGINE      = SearchEngine(FILMS)
GENRES      = all_genres(FILMS)

try:
    from evaluasi_data import EVAL_SUMMARY, EVAL_TFIDF
except Exception:
    EVAL_SUMMARY, EVAL_TFIDF = None, []

print(f"✅ {len(FILMS)} film dimuat.")

# ============================================================
# POSTER HELPER
# ============================================================
_poster_cache: dict = {}

def get_poster_url(film: dict) -> str:
    film_id = film.get("id")
    if film_id in _poster_cache:
        return _poster_cache[film_id]

    for field in ("poster_url", "poster_path"):
        val = film.get(field)
        if val:
            val = str(val).strip()
            if val.startswith("http"):
                _poster_cache[film_id] = val
                return val
            if val.startswith("/"):
                url = f"{TMDB_IMAGE_BASE}{val}"
                _poster_cache[film_id] = url
                return url

    if film_id is not None:
        try:
            api_url = f"{TMDB_API_BASE}/{int(film_id)}?api_key={TMDB_API_KEY}&language=en-US"
            with urllib.request.urlopen(api_url, timeout=8) as resp:
                data = json.load(resp)
            poster_path = data.get("poster_path")
            if poster_path:
                url = f"{TMDB_IMAGE_BASE}{poster_path}"
                _poster_cache[film_id] = url
                return url
        except Exception:
            pass

    _poster_cache[film_id] = PLACEHOLDER_IMG
    return PLACEHOLDER_IMG


def enrich_with_poster(film_dict: dict) -> dict:
    result = dict(film_dict)
    result["poster_url"] = get_poster_url(film_dict)
    return result

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def beranda():
    # Semua film, diurutkan popularitas/rating tertinggi
    all_sorted = sorted(
        FILMS,
        key=lambda f: f.get("popularitas") or f.get("rating") or 0,
        reverse=True
    )
 
    # 4 film teratas untuk hero slideshow
    film_hero  = [enrich_with_poster(f) for f in all_sorted[:4]]
 
    # SEMUA film untuk grid, rows, sidebar — tanpa batas
    film_semua = [enrich_with_poster(f) for f in all_sorted]
 
    return render_template(
        "beranda.html",
        total        = len(FILMS),
        genres       = GENRES,
        film_populer = film_hero,   # hero + Trends Now (tetap pakai nama lama agar bagian lain tidak rusak)
        film_semua   = film_semua,  # SEMUA film untuk grid besar
    )
 



@app.route("/cari")
def cari():
    query        = request.args.get("q", "").strip()
    filter_genre = request.args.get("genre", "")
    top_k        = int(request.args.get("topk", 10))
    hasil        = []
    error        = None

    if query:
        # ── Mode IR: ada query → jalankan search engine ──
        try:
            hits = ENGINE.search(query, top_k=top_k * 3)
            if filter_genre and filter_genre not in ("semua", ""):
                hits = [(f, s) for f, s in hits
                        if filter_genre.lower() in str(f.get("genre", "")).lower()]
            hits = hits[:top_k]
            for film, skor in hits:
                row = enrich_with_poster(film)
                row["skor_relevansi"] = skor
                hasil.append(row)
        except Exception as e:
            error = f"Error saat pencarian: {str(e)}"

    elif filter_genre and filter_genre not in ("semua", ""):
        # ── Mode Browse: tidak ada query, tapi ada genre ──
        # Tampilkan semua film yang punya genre ini, diurutkan rating
        matched = [
            f for f in FILMS
            if filter_genre.lower() in str(f.get("genre", "")).lower()
        ]
        matched.sort(key=lambda f: f.get("rating") or 0, reverse=True)
        for film in matched:
            row = enrich_with_poster(film)
            row["skor_relevansi"] = film.get("rating") or 0  # pakai rating sebagai "skor"
            hasil.append(row)

    return render_template("cari.html",
                           query=query,
                           filter_genre=filter_genre,
                           top_k=top_k,
                           genres=GENRES,
                           hasil=hasil,
                           error=error,
                           jumlah=len(hasil))


@app.route("/detail/<int:film_id>")
def detail(film_id):
    film = FILMS_BY_ID.get(film_id)
    if film is None:
        return render_template("404.html"), 404

    film_data  = enrich_with_poster(film)
    serupa_raw = ENGINE.similar(film, n=7)
    serupa     = []
    for f, s in serupa_raw:
        row = enrich_with_poster(f)
        row["skor_relevansi"] = s
        serupa.append(row)

    return render_template("detail.html", film=film_data, serupa=serupa)


@app.route("/evaluasi")
def evaluasi():
    return render_template("evaluasi.html",
                           summary=EVAL_SUMMARY,
                           tfidf_ev=EVAL_TFIDF)


@app.route("/tentang")
def tentang():
    return render_template("tentang.html",
                           total=len(FILMS),
                           genres=GENRES)


# ============================================================
# API
# ============================================================
@app.route("/api/cari")
def api_cari():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("topk", 10))
    if not query:
        return jsonify({"hasil": [], "jumlah": 0, "error": "Query kosong"})
    try:
        hits  = ENGINE.search(query, top_k=top_k)
        clean = []
        for f, s in hits:
            row = {k: (int(v) if isinstance(v, float) and v == int(v) else v)
                   for k, v in f.items()}
            row["skor_relevansi"] = round(float(s), 6)
            row["poster_url"]     = get_poster_url(f)
            clean.append(row)
        return jsonify({"hasil": clean, "jumlah": len(clean), "error": None})
    except Exception as e:
        return jsonify({"hasil": [], "jumlah": 0, "error": str(e)})


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)