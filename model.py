import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from preprocessing import preprocessing_lengkap

# ============================================================
# MODEL TF-IDF + COSINE SIMILARITY
# ============================================================

class ModelTFIDF:
    def __init__(self):
        self.vectorizer   = TfidfVectorizer()
        self.tfidf_matrix = None
        self.df           = None

    def latih(self, df):
        self.df = df.reset_index(drop=True)
        print("🔄 Melatih model TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(df["teks_preprocessed"])
        print(f"✅ TF-IDF siap! Matrix: {self.tfidf_matrix.shape}")

    def cari(self, query, top_k=10):
        query_bersih = preprocessing_lengkap(query)
        query_vec    = self.vectorizer.transform([query_bersih])
        skor         = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        indeks       = np.argsort(skor)[::-1][:top_k]
        return self._format_hasil(indeks, skor)

    def _format_hasil(self, indeks, skor):
        hasil = []
        for idx in indeks:
            if skor[idx] > 0:
                hasil.append({
                    "id"             : self.df.iloc[idx]["id"],
                    "judul"          : self.df.iloc[idx]["judul"],
                    "sinopsis"       : self.df.iloc[idx]["sinopsis"],
                    "genre"          : self.df.iloc[idx]["genre"],
                    "sutradara"      : self.df.iloc[idx].get("sutradara", ""),
                    "pemain"         : self.df.iloc[idx].get("pemain", ""),
                    "rating"         : self.df.iloc[idx]["rating"],
                    "tahun"          : self.df.iloc[idx]["tahun"],
                    "skor_relevansi" : round(float(skor[idx]), 4)
                })
        return hasil

    def simpan(self, path="data/model_tfidf.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model TF-IDF disimpan: {path}")

    @staticmethod
    def muat(path="data/model_tfidf.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================
# MODEL BM25
# ============================================================

class ModelBM25:
    def __init__(self):
        self.model = None
        self.df    = None

    def latih(self, df):
        self.df = df.reset_index(drop=True)
        print("🔄 Melatih model BM25...")
        korpus     = [teks.split() for teks in df["teks_preprocessed"]]
        self.model = BM25Okapi(korpus)
        print(f"✅ BM25 siap! Total dokumen: {len(korpus)}")

    def cari(self, query, top_k=10):
        query_bersih  = preprocessing_lengkap(query)
        query_tokens  = query_bersih.split()
        skor          = self.model.get_scores(query_tokens)
        indeks        = np.argsort(skor)[::-1][:top_k]
        return self._format_hasil(indeks, skor)

    def _format_hasil(self, indeks, skor):
        hasil = []
        for idx in indeks:
            if skor[idx] > 0:
                hasil.append({
                    "id"             : self.df.iloc[idx]["id"],
                    "judul"          : self.df.iloc[idx]["judul"],
                    "sinopsis"       : self.df.iloc[idx]["sinopsis"],
                    "genre"          : self.df.iloc[idx]["genre"],
                    "sutradara"      : self.df.iloc[idx].get("sutradara", ""),
                    "pemain"         : self.df.iloc[idx].get("pemain", ""),
                    "rating"         : self.df.iloc[idx]["rating"],
                    "tahun"          : self.df.iloc[idx]["tahun"],
                    "skor_relevansi" : round(float(skor[idx]), 4)
                })
        return hasil

    def simpan(self, path="data/model_bm25.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model BM25 disimpan: {path}")

    @staticmethod
    def muat(path="data/model_bm25.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================
# LATIH & SIMPAN KEDUA MODEL
# ============================================================

if __name__ == "__main__":
    print("📂 Membaca dataset preprocessed...")
    df = pd.read_csv("data/dataset_preprocessed.csv")
    print(f"✅ {len(df)} film siap dilatih\n")

    os.makedirs("data", exist_ok=True)

    tfidf = ModelTFIDF()
    tfidf.latih(df)
    tfidf.simpan()

    print()

    bm25 = ModelBM25()
    bm25.latih(df)
    bm25.simpan()

    # Test pencarian
    query_test = "film horor menegangkan Indonesia"
    print(f"\n🔍 Test query: '{query_test}'")

    print("\n--- Hasil TF-IDF ---")
    for i, h in enumerate(tfidf.cari(query_test, top_k=3), 1):
        print(f"  {i}. {h['judul']} | {h['genre']} | Skor: {h['skor_relevansi']}")

    print("\n--- Hasil BM25 ---")
    for i, h in enumerate(bm25.cari(query_test, top_k=3), 1):
        print(f"  {i}. {h['judul']} | {h['genre']} | Skor: {h['skor_relevansi']}")