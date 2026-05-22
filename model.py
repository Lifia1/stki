import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocessing_lengkap


class ModelTFIDF:
    def __init__(self):
        self.vectorizer   = TfidfVectorizer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        self.tfidf_matrix = None
        self.df           = None

    def latih(self, df):
        self.df = df.reset_index(drop=True)
        print("🔄 Melatih model TF-IDF...")
        self.tfidf_matrix = self.vectorizer.fit_transform(df["teks_preprocessed"])
        print(f"✅ TF-IDF siap! Matrix: {self.tfidf_matrix.shape}")
        print(f"   Jumlah term unik: {len(self.vectorizer.vocabulary_)}")

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


if __name__ == "__main__":
    print("📂 Membaca dataset preprocessed...")
    df = pd.read_csv("data/dataset_preprocessed.csv")
    print(f"✅ {len(df)} film siap dilatih\n")

    os.makedirs("data", exist_ok=True)

    tfidf = ModelTFIDF()
    tfidf.latih(df)
    tfidf.simpan()

    print("\n🔍 Test pencarian: 'film horor menegangkan Indonesia'")
    for i, h in enumerate(tfidf.cari("film horor menegangkan Indonesia", top_k=3), 1):
        print(f"  {i}. {h['judul']} | {h['genre']} | Skor: {h['skor_relevansi']}")