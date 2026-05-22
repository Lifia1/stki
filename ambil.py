import requests
import pandas as pd
import time
import os
from tqdm import tqdm

# ============================================================
# KONFIGURASI - Ganti dengan API Key kamu
# ============================================================
API_KEY = "222d708c3712f86a090bc1143ef6ae81"
BASE_URL = "https://api.themoviedb.org/3"
LANGUAGE = "id-ID"
TOTAL_HALAMAN = 300
MIN_KATA_DOKUMEN = 100  # minimal 100 kata dari gabungan semua kolom

def ambil_genre():
    url = f"{BASE_URL}/genre/movie/list"
    params = {"api_key": API_KEY, "language": LANGUAGE}
    response = requests.get(url, params=params)
    data = response.json()
    genre_dict = {g["id"]: g["name"] for g in data["genres"]}
    return genre_dict

def ambil_film_populer(halaman=1):
    url = f"{BASE_URL}/movie/popular"
    params = {"api_key": API_KEY, "language": LANGUAGE, "page": halaman}
    response = requests.get(url, params=params)
    return response.json().get("results", [])

def ambil_detail_film(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY, "language": LANGUAGE}
    response = requests.get(url, params=params)
    return response.json()

def ambil_credits_film(movie_id):
    """Ambil data pemain dan sutradara"""
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": API_KEY, "language": LANGUAGE}
    response = requests.get(url, params=params)
    return response.json()

def gabungkan_teks_dokumen(judul, genre, sinopsis, tagline, sutradara, pemain, produksi, negara):
    """Gabungkan semua kolom menjadi satu dokumen teks lengkap"""
    bagian = []
    if judul:
        bagian.append(f"Judul film {judul}.")
    if genre:
        bagian.append(f"Genre film ini adalah {genre}.")
    if sinopsis:
        bagian.append(sinopsis)
    if tagline:
        bagian.append(f"Tagline: {tagline}.")
    if sutradara:
        bagian.append(f"Disutradarai oleh {sutradara}.")
    if pemain:
        bagian.append(f"Dibintangi oleh {pemain}.")
    if produksi:
        bagian.append(f"Diproduksi oleh {produksi}.")
    if negara:
        bagian.append(f"Negara produksi: {negara}.")
    return " ".join(bagian)

def kumpulkan_data():
    print("🎬 Memulai pengambilan data film dari TMDB API...")
    print("=" * 50)

    print("📂 Mengambil data genre...")
    genre_dict = ambil_genre()
    print(f"✅ {len(genre_dict)} genre berhasil diambil")

    daftar_film = []
    id_film_sudah_ada = set()

    print(f"\n🔄 Mengambil film populer ({TOTAL_HALAMAN} halaman)...")
    for halaman in tqdm(range(1, TOTAL_HALAMAN + 1)):
        films = ambil_film_populer(halaman)
        for film in films:
            if film["id"] not in id_film_sudah_ada:
                id_film_sudah_ada.add(film["id"])
                daftar_film.append(film)
        time.sleep(0.25)

    print(f"✅ {len(daftar_film)} film ditemukan")
    print(f"\n📝 Mengambil detail lengkap setiap film...")
    print(f"   (termasuk tagline, sutradara, pemain, produksi)")

    data_final = []
    dilewati = 0

    for film in tqdm(daftar_film):
        try:
            # Ambil detail film
            detail = ambil_detail_film(film["id"])
            sinopsis  = detail.get("overview", "").strip()
            tagline   = detail.get("tagline", "").strip()

            # Ambil nama genre
            genre_ids = film.get("genre_ids", [])
            nama_genre = ", ".join([genre_dict.get(gid, "") for gid in genre_ids])

            # Ambil perusahaan produksi
            produksi_list = detail.get("production_companies", [])
            produksi = ", ".join([p["name"] for p in produksi_list[:3]])

            # Ambil negara produksi
            negara_list = detail.get("production_countries", [])
            negara = ", ".join([n["name"] for n in negara_list[:2]])

            # Ambil credits (sutradara & pemain)
            time.sleep(0.1)
            credits = ambil_credits_film(film["id"])

            # Sutradara
            crew = credits.get("crew", [])
            sutradara_list = [c["name"] for c in crew if c["job"] == "Director"]
            sutradara = ", ".join(sutradara_list[:2])

            # Pemain utama (top 5)
            cast = credits.get("cast", [])
            pemain_list = [c["name"] for c in cast[:5]]
            pemain = ", ".join(pemain_list)

            # Gabungkan semua jadi satu dokumen
            judul = film.get("title", "")
            teks_dokumen = gabungkan_teks_dokumen(
                judul, nama_genre, sinopsis, tagline,
                sutradara, pemain, produksi, negara
            )

            # Filter: minimal 100 kata dari dokumen gabungan
            jumlah_kata_dokumen = len(teks_dokumen.split())
            if jumlah_kata_dokumen < MIN_KATA_DOKUMEN:
                dilewati += 1
                continue

            data_film = {
                "id"                  : film["id"],
                "judul"               : judul,
                "judul_asli"          : film.get("original_title", ""),
                "sinopsis"            : sinopsis,
                "tagline"             : tagline,
                "genre"               : nama_genre,
                "genre_ids"           : str(genre_ids),
                "sutradara"           : sutradara,
                "pemain"              : pemain,
                "produksi"            : produksi,
                "negara"              : negara,
                "rating"              : film.get("vote_average", 0),
                "jumlah_vote"         : film.get("vote_count", 0),
                "popularitas"         : film.get("popularity", 0),
                "tanggal_rilis"       : film.get("release_date", ""),
                "tahun"               : film.get("release_date", "")[:4] if film.get("release_date") else "",
                "bahasa_asli"         : film.get("original_language", ""),
                "teks_dokumen"        : teks_dokumen,   # ← kolom utama untuk STKI
                "jumlah_kata_dokumen" : jumlah_kata_dokumen,
                "jumlah_kata_sinopsis": len(sinopsis.split())
            }
            data_final.append(data_film)
            time.sleep(0.1)

        except Exception as e:
            continue

    print(f"\n⚠️  Film dilewati (< {MIN_KATA_DOKUMEN} kata): {dilewati}")
    return data_final

def simpan_ke_csv(data, nama_file="data/dataset_film.csv"):
    df = pd.DataFrame(data)
    df = df.sort_values("popularitas", ascending=False).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(nama_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 50}")
    print(f"✅ Dataset berhasil disimpan!")
    print(f"📁 File        : {nama_file}")
    print(f"🎬 Total film  : {len(df)}")
    print(f"\n📈 Statistik kata per dokumen:")
    print(f"   Rata-rata   : {df['jumlah_kata_dokumen'].mean():.0f} kata")
    print(f"   Minimum     : {df['jumlah_kata_dokumen'].min()} kata")
    print(f"   Maximum     : {df['jumlah_kata_dokumen'].max()} kata")
    print(f"\n📈 Statistik sinopsis saja:")
    print(f"   Rata-rata   : {df['jumlah_kata_sinopsis'].mean():.0f} kata")

    print(f"\n📋 Contoh 3 film pertama:")
    for _, row in df.head(3).iterrows():
        print(f"\n  Judul     : {row['judul']}")
        print(f"  Genre     : {row['genre']}")
        print(f"  Sutradara : {row['sutradara']}")
        print(f"  Pemain    : {row['pemain']}")
        print(f"  Kata      : {row['jumlah_kata_dokumen']} kata")
        print(f"  Dokumen   : {row['teks_dokumen'][:120]}...")

    return df

if __name__ == "__main__":
    if API_KEY == "MASUKKAN_API_KEY_KAMU_DISINI":
        print("❌ ERROR: Harap isi API_KEY dengan API key TMDB kamu!")
        print("   Daftar di: https://www.themoviedb.org/settings/api")
        exit()

    data = kumpulkan_data()
    if data:
        simpan_ke_csv(data)
    else:
        print("❌ Tidak ada data yang berhasil dikumpulkan.")