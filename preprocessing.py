import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('punkt', quiet=True)

# ============================================================
# INISIALISASI SASTRAWI
# ============================================================
print("⚙️  Inisialisasi Sastrawi...")
stemmer_factory   = StemmerFactory()
stemmer           = stemmer_factory.create_stemmer()
stopword_factory  = StopWordRemoverFactory()
stopword_remover  = stopword_factory.create_stop_word_remover()
print("✅ Sastrawi siap digunakan")

# ============================================================
# FUNGSI PREPROCESSING
# ============================================================

def case_folding(teks):
    """Ubah semua huruf menjadi huruf kecil"""
    return teks.lower()

def cleaning(teks):
    """Hapus karakter yang tidak diperlukan"""
    teks = re.sub(r'http\S+|www\S+', '', teks)         # hapus URL
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)           # hapus tanda baca & angka
    teks = re.sub(r'\s+', ' ', teks).strip()            # hapus spasi berlebih
    return teks

def tokenisasi(teks):
    """Pisahkan teks menjadi daftar kata"""
    return teks.split()

def hapus_stopword(teks):
    """Hapus kata-kata yang tidak penting"""
    return stopword_remover.remove(teks)

def stemming(teks):
    """Potong imbuhan kata"""
    return stemmer.stem(teks)

def preprocessing_lengkap(teks):
    """Pipeline preprocessing lengkap untuk satu teks"""
    if not isinstance(teks, str) or teks.strip() == "":
        return ""
    teks = case_folding(teks)
    teks = cleaning(teks)
    teks = hapus_stopword(teks)
    teks = stemming(teks)
    tokens = tokenisasi(teks)
    return " ".join(tokens)

# ============================================================
# PROSES DATASET
# ============================================================

def proses_dataset(
    input_file  = "data/dataset_film.csv",
    output_file = "data/dataset_preprocessed.csv"
):
    print(f"\n📂 Membaca dataset: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ {len(df)} film berhasil dibaca")

    print("\n🔄 Melakukan preprocessing teks_dokumen...")
    hasil = []

    for i, row in df.iterrows():
        if i % 50 == 0:
            print(f"   Proses {i}/{len(df)}...")

        # Preprocessing kolom teks_dokumen (gabungan semua info)
        teks_bersih = preprocessing_lengkap(str(row.get("teks_dokumen", "")))

        hasil.append({
            "id"                  : row["id"],
            "judul"               : row["judul"],
            "sinopsis"            : row["sinopsis"],
            "tagline"             : row.get("tagline", ""),
            "genre"               : row["genre"],
            "sutradara"           : row.get("sutradara", ""),
            "pemain"              : row.get("pemain", ""),
            "produksi"            : row.get("produksi", ""),
            "negara"              : row.get("negara", ""),
            "rating"              : row["rating"],
            "tahun"               : row["tahun"],
            "bahasa_asli"         : row["bahasa_asli"],
            "teks_dokumen"        : row.get("teks_dokumen", ""),
            "teks_preprocessed"   : teks_bersih,
            "jumlah_kata_dokumen" : row.get("jumlah_kata_dokumen", 0)
        })

    df_hasil = pd.DataFrame(hasil)
    # Hapus yang teks preprocessed-nya kosong
    df_hasil = df_hasil[df_hasil["teks_preprocessed"].str.strip() != ""].reset_index(drop=True)
    df_hasil.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 50}")
    print(f"✅ Preprocessing selesai!")
    print(f"📁 File           : {output_file}")
    print(f"🎬 Total film     : {len(df_hasil)}")
    print(f"\n📋 Contoh hasil preprocessing:")
    for _, row in df_hasil.head(2).iterrows():
        print(f"\n  Judul        : {row['judul']}")
        print(f"  Teks asli    : {str(row['teks_dokumen'])[:100]}...")
        print(f"  Preprocessed : {row['teks_preprocessed'][:100]}...")

    return df_hasil

if __name__ == "__main__":
    proses_dataset()