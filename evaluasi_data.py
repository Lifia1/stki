"""Ringkasan & hasil evaluasi per query (Precision@K, Recall@K, AP, NDCG@K)."""

EVAL_SUMMARY = {
    "TF-IDF": {"MAP": 0.2375, "NDCG": 0.6979, "Precision": 0.68, "Recall": 0.1828},
    "BM25":   {"MAP": 0.2535, "NDCG": 0.6867, "Precision": 0.62, "Recall": 0.1664},
}

EVAL_TFIDF = [
    {"no": 1, "query": "film superhero aksi petualangan Marvel DC", "precision_at_k": 1.0, "recall_at_k": 0.1923, "ap": 0.3277, "ndcg_at_k": 1.0},
    {"no": 2, "query": "film sihir fantasi petualangan penyihir dunia sihir", "precision_at_k": 1.0, "recall_at_k": 0.3333, "ap": 0.5333, "ndcg_at_k": 1.0},
    {"no": 3, "query": "film horor seram menegangkan hantu monster zombie", "precision_at_k": 0.6, "recall_at_k": 0.1765, "ap": 0.1898, "ndcg_at_k": 0.6164},
    {"no": 4, "query": "film animasi lucu keluarga anak-anak kartun", "precision_at_k": 0.2, "recall_at_k": 0.04, "ap": 0.0289, "ndcg_at_k": 0.214},
    {"no": 5, "query": "film aksi laga kriminal penjahat polisi kejahatan", "precision_at_k": 0.8, "recall_at_k": 0.1739, "ap": 0.2341, "ndcg_at_k": 0.8539},
    {"no": 6, "query": "film fiksi ilmiah luar angkasa teknologi masa depan robot", "precision_at_k": 0.4, "recall_at_k": 0.1053, "ap": 0.114, "ndcg_at_k": 0.5087},
    {"no": 7, "query": "film misteri detektif thriller investigasi pembunuhan", "precision_at_k": 0.8, "recall_at_k": 0.2857, "ap": 0.2744, "ndcg_at_k": 0.786},
    {"no": 8, "query": "film petualangan alam liar bertahan hidup ekspedisi", "precision_at_k": 0.2, "recall_at_k": 0.0667, "ap": 0.05, "ndcg_at_k": 0.214},
    {"no": 9, "query": "film komedi humor ringan lucu menghibur", "precision_at_k": 0.8, "recall_at_k": 0.16, "ap": 0.2183, "ndcg_at_k": 0.786},
    {"no": 10, "query": "film drama percintaan romantis cinta mengharukan", "precision_at_k": 1.0, "recall_at_k": 0.2941, "ap": 0.4044, "ndcg_at_k": 1.0},
]

EVAL_BM25 = [
    {"no": 1, "query": "film superhero aksi petualangan Marvel DC", "precision_at_k": 0.8, "recall_at_k": 0.1538, "ap": 0.304, "ndcg_at_k": 0.8304},
    {"no": 2, "query": "film sihir fantasi petualangan penyihir dunia sihir", "precision_at_k": 1.0, "recall_at_k": 0.3333, "ap": 0.5333, "ndcg_at_k": 1.0},
    {"no": 3, "query": "film horor seram menegangkan hantu monster zombie", "precision_at_k": 0.4, "recall_at_k": 0.1176, "ap": 0.2486, "ndcg_at_k": 0.5531},
    {"no": 4, "query": "film animasi lucu keluarga anak-anak kartun", "precision_at_k": 0.4, "recall_at_k": 0.08, "ap": 0.0667, "ndcg_at_k": 0.5087},
    {"no": 5, "query": "film aksi laga kriminal penjahat polisi kejahatan", "precision_at_k": 0.8, "recall_at_k": 0.1739, "ap": 0.2732, "ndcg_at_k": 0.8688},
    {"no": 6, "query": "film fiksi ilmiah luar angkasa teknologi masa depan robot", "precision_at_k": 0.4, "recall_at_k": 0.1053, "ap": 0.1316, "ndcg_at_k": 0.5531},
    {"no": 7, "query": "film misteri detektif thriller investigasi pembunuhan", "precision_at_k": 0.8, "recall_at_k": 0.2857, "ap": 0.3732, "ndcg_at_k": 0.8688},
    {"no": 8, "query": "film petualangan alam liar bertahan hidup ekspedisi", "precision_at_k": 0.0, "recall_at_k": 0.0, "ap": 0.0111, "ndcg_at_k": 0.0},
    {"no": 9, "query": "film komedi humor ringan lucu menghibur", "precision_at_k": 0.6, "recall_at_k": 0.12, "ap": 0.1892, "ndcg_at_k": 0.6844},
    {"no": 10, "query": "film drama percintaan romantis cinta mengharukan", "precision_at_k": 1.0, "recall_at_k": 0.2941, "ap": 0.4044, "ndcg_at_k": 1.0},
]
