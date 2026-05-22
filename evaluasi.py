import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from model import ModelTFIDF, ModelBM25

# ============================================================
# 10 QUERY PENGUJIAN + GROUND TRUTH
# Disesuaikan dengan dataset 183 film
# ============================================================

QUERY_GROUND_TRUTH = [
    {
        "no"     : 1,
        "query"  : "film superhero aksi petualangan Marvel DC",
        "relevan": [
            "The Avengers", "Avengers: Infinity War", "Avengers: Age of Ultron",
            "Iron Man", "Iron Man 2", "Iron Man 3",
            "Black Widow", "Kapten Amerika: Avenger Pertama",
            "Thor: Cinta dan Guntur", "Thor: The Dark World",
            "Doctor Strange", "Spider-Man: Brand New Day",
            "The Amazing Spider-Man", "The Amazing Spider-Man 2",
            "Ant-Man and the Wasp: Quantumania",
            "X-Men: Dark Phoenix", "Fantastic Four",
            "Venom", "Venom: The Last Dance",
            "Deadpool", "Logan",
            "The Dark Knight Rises", "Man of Steel",
            "Suicide Squad", "Hellboy", "Hellboy II: The Golden Army",
        ]
    },
    {
        "no"     : 2,
        "query"  : "film sihir fantasi petualangan penyihir dunia sihir",
        "relevan": [
            "Harry Potter dan Batu Bertuah",
            "Harry Potter and the Prisoner of Azkaban",
            "Harry Potter and the Goblet of Fire",
            "Harry Potter and the Half-Blood Prince",
            "Hewan-Hewan Fantastis dan Dimana Mereka Bisa Ditemukan",
            "Fantastic Beasts: The Secrets of Dumbledore",
            "Hewan Fantastis: Kejahatan Grindelwald",
            "The Lord of the Rings: The Fellowship of the Ring",
            "The Lord of the Rings: The War of the Rohirrim",
            "The Hobbit: The Battle of the Five Armies",
            "The Sorcerer's Apprentice",
            "Maleficent", "Snow White",
            "The Condor Heroes",
            "Journey to the West: The Demons Strike Back",
        ]
    },
    {
        "no"     : 3,
        "query"  : "film horor seram menegangkan hantu monster zombie",
        "relevan": [
            "The Nun", "Split", "Sinister", "The Others",
            "A Quiet Place Part II",
            "Resident Evil: Welcome to Raccoon City",
            "Dawn of the Dead", "Insidious: The Red Door",
            "Don't Breathe 2", "Abigail",
            "반도", "Annabelle Comes Home",
            "Halloween Ends", "Ouija: Origin of Evil",
            "Zombieland: Double Tap", "Lembayung",
            "Old",
        ]
    },
    {
        "no"     : 4,
        "query"  : "film animasi lucu keluarga anak-anak kartun",
        "relevan": [
            "Zootopia 2", "The Super Mario Bros. Movie",
            "Brave", "Madagascar", "Madagascar: Escape 2 Africa",
            "Ice Age: The Meltdown", "Ice Age: Collision Course",
            "Soul", "Cars 3", "The Lion King",
            "Incredibles 2", "Raya and the Last Dragon",
            "Masuk Keluar", "Corpse Bride",
            "Suzume no Tojimari", "Jujutsu Kaisen 0",
            "ONE PIECE FILM RED", "The SpongeBob Movie: Sponge Out of Water",
            "Trolls", "Tom & Jerry", "Scoob!",
            "Hoodwinked!", "Legend of the Guardians: The Owls of Ga'Hoole",
            "Doraemon: Petualangan Nobita di Pulau Harta Karun",
            "The Lord of the Rings: The War of the Rohirrim",
        ]
    },
    {
        "no"     : 5,
        "query"  : "film aksi laga kriminal penjahat polisi kejahatan",
        "relevan": [
            "The Fate of the Furious", "F9: The Fast Saga",
            "Fast & Furious Presents: Hobbs & Shaw",
            "Wrath of Man", "Rush Hour 2", "Rush Hour 3",
            "Die Hard 2", "Die Hard: With a Vengeance",
            "A Good Day to Die Hard",
            "Jack Reacher: Never Go Back",
            "The Man From Nowhere", "New World",
            "Believer", "Special Delivery", "Unstoppable",
            "Memory", "RoboCop",
            "The Dark Knight Rises",
            "Snake Eyes: G.I. Joe Origins",
            "Johnny English Strikes Again",
            "Johnny English Reborn",
            "Hitman's Wife's Bodyguard",
            "Mile 22",
        ]
    },
    {
        "no"     : 6,
        "query"  : "film fiksi ilmiah luar angkasa teknologi masa depan robot",
        "relevan": [
            "Dune", "Arrival",
            "Avatar: The Way of Water",
            "Star Wars: Episode V - The Empire Strikes Back",
            "The Matrix Reloaded",
            "Terminator: Dark Fate",
            "Twelve Monkeys",
            "Kerajaan Planet Kera",
            "65", "Valerian and the City of a Thousand Planets",
            "The Predator", "The Adam Project",
            "Men in Black", "Men in Black II", "Men in Black 3",
            "RoboCop", "Cloud Atlas",
            "Wifelike", "원더랜드",
        ]
    },
    {
        "no"     : 7,
        "query"  : "film misteri detektif thriller investigasi pembunuhan",
        "relevan": [
            "Basic Instinct", "Old Boy", "Sinister",
            "The Others", "Enola Holmes", "Inferno",
            "Molly's Game", "The Accountant",
            "Searching", "Mindcage",
            "Old", "Twelve Monkeys",
            "The Devil All the Time",
            "Dogville",
        ]
    },
    {
        "no"     : 8,
        "query"  : "film petualangan alam liar bertahan hidup ekspedisi",
        "relevan": [
            "Apocalypto", "The Revenant",
            "Jungle Cruise",
            "Jumanji: Welcome to the Jungle",
            "Jumanji: The Next Level",
            "Hacksaw Ridge", "Logan", "Fall", "65",
            "Alpha", "Love and Monsters",
            "The 13th Warrior",
            "Thirteen Lives",
            "Those Who Wish Me Dead",
            "Escape from Pretoria",
        ]
    },
    {
        "no"     : 9,
        "query"  : "film komedi humor ringan lucu menghibur",
        "relevan": [
            "Rush Hour 2", "Rush Hour 3",
            "Men in Black", "Men in Black II", "Men in Black 3",
            "Deadpool", "Cruella", "Notting Hill",
            "Fast & Furious Presents: Hobbs & Shaw",
            "Hitman's Wife's Bodyguard",
            "Ghostbusters: Frozen Empire",
            "Sonic the Hedgehog 2",
            "The Adam Project",
            "To All the Boys I've Loved Before",
            "Crazy Rich Asians",
            "Tom & Jerry", "Hustle",
            "The Unbearable Weight of Massive Talent",
            "Bridget Jones's Baby",
            "Kung Fu Yoga",
            "Johnny English Strikes Again",
            "Johnny English Reborn",
            "Dora and the Lost City of Gold",
            "The Banshees of Inisherin",
            "賭俠 III 之上海灘賭聖",
        ]
    },
    {
        "no"     : 10,
        "query"  : "film drama percintaan romantis cinta mengharukan",
        "relevan": [
            "Notting Hill", "Culpa Mía: London",
            "Bermain dengan Cinta",
            "The Twilight Saga: Eclipse",
            "Corpse Bride",
            "The Greatest Showman",
            "Soul", "12 Angry Men",
            "Maleficent",
            "To All the Boys I've Loved Before",
            "Crazy Rich Asians",
            "A Star Is Born",
            "Rocketman",
            "Bridget Jones's Baby",
            "Werk ohne Autor",
            "딸의 친구",
            "Drive My Car",
        ]
    },
]

# ============================================================
# FUNGSI EVALUASI
# ============================================================

def cek_relevan(judul_hasil, ground_truth):
    judul_lower = judul_hasil.lower().strip()
    for gt in ground_truth:
        gt_lower = gt.lower().strip()
        if gt_lower in judul_lower or judul_lower in gt_lower:
            return True
    return False

def precision_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    top_k   = hasil[:k]
    relevan = sum(1 for h in top_k if cek_relevan(h["judul"], ground_truth))
    return relevan / k

def recall_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    top_k   = hasil[:k]
    relevan = sum(1 for h in top_k if cek_relevan(h["judul"], ground_truth))
    return relevan / len(ground_truth)

def average_precision(hasil, ground_truth):
    if not hasil or not ground_truth:
        return 0.0
    total_ap = 0.0
    hit      = 0
    for i, h in enumerate(hasil, 1):
        if cek_relevan(h["judul"], ground_truth):
            hit      += 1
            total_ap += hit / i
    return total_ap / len(ground_truth)

def ndcg_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    dcg = 0.0
    for i, h in enumerate(hasil[:k], 1):
        rel  = 1 if cek_relevan(h["judul"], ground_truth) else 0
        dcg += rel / np.log2(i + 1)
    ideal = sum(1 / np.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0

# ============================================================
# JALANKAN EVALUASI
# ============================================================

def evaluasi_model(model, nama_model, k=5):
    print(f"\n{'=' * 60}")
    print(f"📊 Evaluasi Model: {nama_model}")
    print(f"{'=' * 60}")

    hasil_per_query = []

    for item in QUERY_GROUND_TRUTH:
        query  = item["query"]
        gt     = item["relevan"]
        hasil  = model.cari(query, top_k=10)

        p_at_k = precision_at_k(hasil, gt, k)
        r_at_k = recall_at_k(hasil, gt, k)
        ap     = average_precision(hasil, gt)
        ndcg   = ndcg_at_k(hasil, gt, k)

        hasil_per_query.append({
            "no"            : item["no"],
            "query"         : query,
            "precision_at_k": round(p_at_k, 4),
            "recall_at_k"   : round(r_at_k, 4),
            "ap"            : round(ap, 4),
            "ndcg_at_k"     : round(ndcg, 4),
        })

        top3 = ", ".join([h["judul"] for h in hasil[:3]]) if hasil else "(tidak ada)"
        print(f"\n  Q{item['no']}  : {query}")
        print(f"  Top-3   : {top3}")
        print(f"  P@{k}: {p_at_k:.4f} | R@{k}: {r_at_k:.4f} | AP: {ap:.4f} | NDCG@{k}: {ndcg:.4f}")

    df_eval   = pd.DataFrame(hasil_per_query)
    map_score = df_eval["ap"].mean()
    mean_ndcg = df_eval["ndcg_at_k"].mean()
    mean_prec = df_eval["precision_at_k"].mean()
    mean_rec  = df_eval["recall_at_k"].mean()

    print(f"\n{'─' * 60}")
    print(f"  📌 MAP (Mean Average Precision) : {map_score:.4f}")
    print(f"  📌 Mean NDCG@{k}                 : {mean_ndcg:.4f}")
    print(f"  📌 Mean Precision@{k}            : {mean_prec:.4f}")
    print(f"  📌 Mean Recall@{k}               : {mean_rec:.4f}")
    print(f"{'─' * 60}")

    return df_eval, {
        "MAP"      : round(map_score, 4),
        "NDCG"     : round(mean_ndcg, 4),
        "Precision": round(mean_prec, 4),
        "Recall"   : round(mean_rec, 4)
    }

# ============================================================
# SIMPAN HASIL & GRAFIK
# ============================================================

def simpan_hasil_evaluasi(df_tfidf, df_bm25, summary_tfidf, summary_bm25):
    os.makedirs("data", exist_ok=True)

    df_tfidf.to_csv("data/evaluasi_tfidf.csv", index=False, encoding="utf-8-sig")
    df_bm25.to_csv("data/evaluasi_bm25.csv",   index=False, encoding="utf-8-sig")

    summary = {"TF-IDF": summary_tfidf, "BM25": summary_bm25}
    with open("data/evaluasi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Hasil evaluasi disimpan di folder data/")

    # ── Grafik 1: Perbandingan metrik ──
    metrik    = ["MAP", "NDCG", "Precision", "Recall"]
    val_tfidf = [summary_tfidf[m] for m in metrik]
    val_bm25  = [summary_bm25[m]  for m in metrik]
    x, width  = np.arange(len(metrik)), 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, val_tfidf, width, label="TF-IDF", color="#4C72B0")
    bars2 = ax.bar(x + width/2, val_bm25,  width, label="BM25",   color="#DD8452")
    ax.set_title("Perbandingan Performa TF-IDF vs BM25", fontsize=14, fontweight="bold")
    ax.set_ylabel("Skor")
    ax.set_xticks(x)
    ax.set_xticklabels(metrik)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig("data/grafik_perbandingan.png", dpi=150)
    plt.close()

    # ── Grafik 2: AP per query ──
    qlabel = [f"Q{i+1}" for i in range(len(df_tfidf))]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(qlabel, df_tfidf["ap"], marker="o", label="TF-IDF", color="#4C72B0", linewidth=2)
    ax.plot(qlabel, df_bm25["ap"],  marker="s", label="BM25",   color="#DD8452", linewidth=2)
    ax.set_title("Average Precision per Query", fontsize=13, fontweight="bold")
    ax.set_ylabel("Average Precision")
    ax.set_xlabel("Query")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    for i, (y1, y2) in enumerate(zip(df_tfidf["ap"], df_bm25["ap"])):
        ax.annotate(f"{y1:.2f}", (qlabel[i], y1), textcoords="offset points", xytext=(-10, 6), fontsize=8, color="#4C72B0")
        ax.annotate(f"{y2:.2f}", (qlabel[i], y2), textcoords="offset points", xytext=(4,   6), fontsize=8, color="#DD8452")
    plt.tight_layout()
    plt.savefig("data/grafik_ap_per_query.png", dpi=150)
    plt.close()

    # ── Grafik 3: Precision@K per query ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df_ev, nama, warna in zip(axes, [df_tfidf, df_bm25], ["TF-IDF", "BM25"], ["#4C72B0", "#DD8452"]):
        ax.bar(qlabel, df_ev["precision_at_k"], color=warna, alpha=0.85)
        ax.set_title(f"Precision@5 per Query — {nama}", fontweight="bold")
        ax.set_ylabel("Precision@5")
        ax.set_ylim(0, 1.1)
        mean_val = df_ev["precision_at_k"].mean()
        ax.axhline(mean_val, color="red", linestyle="--", label=f"Rata-rata: {mean_val:.3f}")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("data/grafik_precision_per_query.png", dpi=150)
    plt.close()

    print("📊 3 grafik disimpan di folder data/")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("📂 Memuat model TF-IDF dan BM25...")
    try:
        tfidf = ModelTFIDF.muat("data/model_tfidf.pkl")
        bm25  = ModelBM25.muat("data/model_bm25.pkl")
    except FileNotFoundError:
        print("❌ Model belum ada! Jalankan dulu: python model.py")
        exit()

    df_tfidf, summary_tfidf = evaluasi_model(tfidf, "TF-IDF + Cosine Similarity", k=5)
    df_bm25,  summary_bm25  = evaluasi_model(bm25,  "BM25",                       k=5)

    simpan_hasil_evaluasi(df_tfidf, df_bm25, summary_tfidf, summary_bm25)

    print(f"\n{'=' * 60}")
    print("🏆 KESIMPULAN AKHIR EVALUASI")
    print(f"{'=' * 60}")
    print(f"  TF-IDF → MAP: {summary_tfidf['MAP']:.4f} | NDCG@5: {summary_tfidf['NDCG']:.4f} | P@5: {summary_tfidf['Precision']:.4f}")
    print(f"  BM25   → MAP: {summary_bm25['MAP']:.4f} | NDCG@5: {summary_bm25['NDCG']:.4f} | P@5: {summary_bm25['Precision']:.4f}")
    pemenang = "TF-IDF" if summary_tfidf["MAP"] >= summary_bm25["MAP"] else "BM25"
    print(f"\n  🥇 Model terbaik berdasarkan MAP: {pemenang}")
    print(f"\n  ✅ Semua hasil disimpan di folder data/")