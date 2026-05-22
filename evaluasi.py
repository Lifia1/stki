import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from model import ModelTFIDF

# ============================================================
# 10 QUERY PENGUJIAN + GROUND TRUTH
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
            "Zombieland: Double Tap", "Lembayung", "Old",
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
            "Hitman's Wife's Bodyguard", "Mile 22",
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
            "RoboCop", "Cloud Atlas", "Wifelike", "원더랜드",
        ]
    },
    {
        "no"     : 7,
        "query"  : "film misteri detektif thriller investigasi pembunuhan",
        "relevan": [
            "Basic Instinct", "Old Boy", "Sinister",
            "The Others", "Enola Holmes", "Inferno",
            "Molly's Game", "The Accountant",
            "Searching", "Mindcage", "Old", "Twelve Monkeys",
            "The Devil All the Time", "Dogville",
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
            "The 13th Warrior", "Thirteen Lives",
            "Those Who Wish Me Dead", "Escape from Pretoria",
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
            "Sonic the Hedgehog 2", "The Adam Project",
            "To All the Boys I've Loved Before",
            "Crazy Rich Asians", "Tom & Jerry", "Hustle",
            "The Unbearable Weight of Massive Talent",
            "Bridget Jones's Baby", "Kung Fu Yoga",
            "Johnny English Strikes Again", "Johnny English Reborn",
            "Dora and the Lost City of Gold",
            "The Banshees of Inisherin",
        ]
    },
    {
        "no"     : 10,
        "query"  : "film drama percintaan romantis cinta mengharukan",
        "relevan": [
            "Notting Hill", "Culpa Mía: London",
            "Bermain dengan Cinta",
            "The Twilight Saga: Eclipse",
            "Corpse Bride", "The Greatest Showman",
            "Soul", "12 Angry Men", "Maleficent",
            "To All the Boys I've Loved Before",
            "Crazy Rich Asians", "A Star Is Born",
            "Rocketman", "Bridget Jones's Baby",
            "Werk ohne Autor", "딸의 친구", "Drive My Car",
        ]
    },
]

# ============================================================
# FUNGSI EVALUASI
# ============================================================

def cek_relevan(judul_hasil, ground_truth):
    judul_lower = judul_hasil.lower().strip()
    for gt in ground_truth:
        if gt.lower().strip() in judul_lower or judul_lower in gt.lower().strip():
            return True
    return False

def precision_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    relevan = sum(1 for h in hasil[:k] if cek_relevan(h["judul"], ground_truth))
    return relevan / k

def recall_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    relevan = sum(1 for h in hasil[:k] if cek_relevan(h["judul"], ground_truth))
    return relevan / len(ground_truth)

def average_precision(hasil, ground_truth):
    if not hasil or not ground_truth:
        return 0.0
    total_ap, hit = 0.0, 0
    for i, h in enumerate(hasil, 1):
        if cek_relevan(h["judul"], ground_truth):
            hit      += 1
            total_ap += hit / i
    return total_ap / len(ground_truth)

def ndcg_at_k(hasil, ground_truth, k):
    if not hasil or not ground_truth:
        return 0.0
    dcg   = sum((1 / np.log2(i + 1)) for i, h in enumerate(hasil[:k], 1)
                if cek_relevan(h["judul"], ground_truth))
    ideal = sum(1 / np.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0

# ============================================================
# JALANKAN EVALUASI
# ============================================================

def evaluasi_model(model, k=5):
    print(f"\n{'=' * 60}")
    print("📊 Evaluasi Model: TF-IDF + Cosine Similarity")
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
            "ap"            : round(ap,     4),
            "ndcg_at_k"     : round(ndcg,   4),
        })

        top3 = ", ".join([h["judul"] for h in hasil[:3]]) if hasil else "(tidak ada)"
        print(f"\n  Q{item['no']:02d} : {query}")
        print(f"  Top-3  : {top3}")
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
        "Recall"   : round(mean_rec,  4),
    }

# ============================================================
# SIMPAN HASIL & GRAFIK
# ============================================================

def simpan_hasil_evaluasi(df_eval, summary):
    os.makedirs("data",   exist_ok=True)
    os.makedirs("static", exist_ok=True)

    df_eval.to_csv("data/evaluasi_tfidf.csv", index=False, encoding="utf-8-sig")

    with open("data/evaluasi_summary.json", "w") as f:
        json.dump({"TF-IDF": summary}, f, indent=2)

    print("\n✅ Hasil evaluasi disimpan di folder data/")

    qlabel = [f"Q{i+1}" for i in range(len(df_eval))]

    # ── Grafik 1: Metrik ringkasan (bar chart) ──
    metrik = ["MAP", "NDCG", "Precision", "Recall"]
    vals   = [summary[m] for m in metrik]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(metrik, vals, color="#839958", width=0.45, zorder=3)
    ax.set_title("Metrik Evaluasi TF-IDF + Cosine Similarity",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Skor")
    ax.set_ylim(0, 1.15)
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f6")
    fig.patch.set_facecolor("#f9f9f6")
    plt.tight_layout()
    plt.savefig("data/grafik_metrik.png",   dpi=150)
    plt.savefig("static/grafik_metrik.png", dpi=150)
    plt.close()

    # ── Grafik 2: AP per query ──
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(qlabel, df_eval["ap"], marker="o", color="#839958",
            linewidth=2, markersize=7, label="AP per Query")
    ax.axhline(summary["MAP"], color="#D3968C", linestyle="--",
               linewidth=1.5, label=f"MAP = {summary['MAP']:.4f}")
    ax.set_title("Average Precision per Query — TF-IDF",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Precision")
    ax.set_xlabel("Query")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#f9f9f6")
    fig.patch.set_facecolor("#f9f9f6")
    for i, y in enumerate(df_eval["ap"]):
        ax.annotate(f"{y:.2f}", (qlabel[i], y),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color="#4a5e2a")
    plt.tight_layout()
    plt.savefig("data/grafik_ap_per_query.png",   dpi=150)
    plt.savefig("static/grafik_ap_per_query.png", dpi=150)
    plt.close()

    # ── Grafik 3: Precision@K per query ──
    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors  = ["#839958" if v >= summary["Precision"] else "#D3968C"
               for v in df_eval["precision_at_k"]]
    ax.bar(qlabel, df_eval["precision_at_k"], color=colors, alpha=0.85, zorder=3)
    mean_p = df_eval["precision_at_k"].mean()
    ax.axhline(mean_p, color="#105666", linestyle="--",
               linewidth=1.5, label=f"Rata-rata P@5 = {mean_p:.4f}")
    ax.set_title("Precision@5 per Query — TF-IDF",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Precision@5")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f6")
    fig.patch.set_facecolor("#f9f9f6")
    plt.tight_layout()
    plt.savefig("data/grafik_precision_per_query.png",   dpi=150)
    plt.savefig("static/grafik_precision_per_query.png", dpi=150)
    plt.close()

    # ── Grafik 4: NDCG@K per query ──
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(qlabel, df_eval["ndcg_at_k"], color="#105666", alpha=0.8, zorder=3)
    ax.axhline(df_eval["ndcg_at_k"].mean(), color="#D3968C", linestyle="--",
               linewidth=1.5, label=f"Rata-rata NDCG@5 = {summary['NDCG']:.4f}")
    ax.set_title("NDCG@5 per Query — TF-IDF",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("NDCG@5")
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_facecolor("#f9f9f6")
    fig.patch.set_facecolor("#f9f9f6")
    plt.tight_layout()
    plt.savefig("data/grafik_ndcg_per_query.png",   dpi=150)
    plt.savefig("static/grafik_ndcg_per_query.png", dpi=150)
    plt.close()

    print("📊 4 grafik disimpan di data/ dan static/")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("📂 Memuat model TF-IDF...")
    try:
        tfidf = ModelTFIDF.muat("data/model_tfidf.pkl")
    except FileNotFoundError:
        print("❌ Model belum ada! Jalankan dulu: python model.py")
        exit()

    df_eval, summary = evaluasi_model(tfidf, k=5)
    simpan_hasil_evaluasi(df_eval, summary)

    print(f"\n{'=' * 60}")
    print("🏆 HASIL AKHIR EVALUASI TF-IDF")
    print(f"{'=' * 60}")
    print(f"  MAP         : {summary['MAP']:.4f}")
    print(f"  NDCG@5      : {summary['NDCG']:.4f}")
    print(f"  Precision@5 : {summary['Precision']:.4f}")
    print(f"  Recall@5    : {summary['Recall']:.4f}")
    print(f"\n  ✅ Semua hasil disimpan di folder data/ dan static/")