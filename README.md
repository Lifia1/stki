# CineFind — Sistem Temu Kembali Informasi Film

Proyek UAS Temu Kembali Informasi (COM620321)  
Universitas Lampung FMIPA · Periode 2025-2026 Genap

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Opsional) Ulangi preprocessing data
```bash
python preprocessing.py
```

### 3. (Opsional) Latih ulang model
```bash
python model.py
```

### 4. (Opsional) Jalankan evaluasi
```bash
python evaluasi.py
```

### 5. Jalankan aplikasi Flask
```bash
python app.py
```

Buka browser: http://localhost:5000

---

## Struktur Proyek
```
stki/
├── app.py               # Flask web application (MAIN)
├── app_streamlit.py     # Versi Streamlit (alternatif)
├── model.py             # Model TF-IDF dan BM25
├── preprocessing.py     # Pipeline preprocessing teks
├── evaluasi.py          # Evaluasi MAP, NDCG, Precision, Recall
├── ambil.py             # Script ambil data dari TMDB API
├── requirements.txt     # Daftar library
├── templates/           # Template HTML Flask
│   ├── base.html
│   ├── beranda.html
│   ├── cari.html
│   ├── detail.html
│   ├── evaluasi.html
│   ├── tentang.html
│   └── 404.html
├── static/              # Aset statis (grafik)
└── data/
    ├── dataset_film.csv             # Dataset mentah
    ├── dataset_preprocessed.csv     # Dataset setelah preprocessing
    ├── model_tfidf.pkl              # Model TF-IDF tersimpan
    ├── model_bm25.pkl               # Model BM25 tersimpan
    ├── evaluasi_tfidf.csv           # Hasil evaluasi TF-IDF
    ├── evaluasi_bm25.csv            # Hasil evaluasi BM25
    └── evaluasi_summary.json        # Ringkasan evaluasi
```

## Fitur Sistem
- Pencarian film dengan query teks bebas
- Dua model IR: TF-IDF + Cosine Similarity dan BM25
- Filter genre dan jumlah hasil
- Halaman detail film beserta film serupa
- Halaman evaluasi dengan metrik MAP, NDCG@5, Precision@5, Recall@5
- API endpoint JSON untuk pencarian
