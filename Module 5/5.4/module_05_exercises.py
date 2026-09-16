# 5.4 Module 05 Exercises
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# 1. Analisis LLM Benchmark Scores (Pandas)
# -------------------------------------------------------------------
def generate_benchmark_csv(csv_path: str | Path) -> None:
    """Membuat file CSV sintetis benchmark LLM dengan minimal 20 baris dan 4 model."""
    models = ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "llama-3-70b"]
    tasks = ["GSM8K Math", "HumanEval Coding", "MMLU Reasoning", "Doc Summarization", "Translation"]

    data = []
    rng = np.random.default_rng(42)
    for model in models:
        for task in tasks:
            score = round(float(rng.uniform(0.60, 0.98)), 3)
            latency = round(float(rng.uniform(150.0, 1200.0)), 1)
            data.append({"model": model, "task": task, "score": score, "latency_ms": latency})

    df = pd.DataFrame(data)
    file_path = Path(csv_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


def analyze_benchmark_scores(csv_path: str | Path) -> dict:
    """
    Menganalisis file CSV benchmark score:
    1. Rata-rata score per model.
    2. Task dengan performa terbaik (score tertinggi) per model.
    3. Korelasi antara kolom 'score' dan 'latency_ms'.
    """
    df = pd.read_csv(csv_path)

    # 1. Mean score per model
    mean_scores = df.groupby("model")["score"].mean().round(4).to_dict()

    # 2. Best performing task per model
    idx_max = df.groupby("model")["score"].idxmax()
    best_tasks = df.loc[idx_max, ["model", "task", "score"]].to_dict(orient="records")

    # 3. Correlation between score and latency_ms
    correlation = float(df["score"].corr(df["latency_ms"]))

    return {
        "mean_score_per_model": mean_scores,
        "best_task_per_model": best_tasks,
        "score_latency_correlation": round(correlation, 4),
    }


# -------------------------------------------------------------------
# 2. L2 Normalization of Embeddings (NumPy)
# -------------------------------------------------------------------
def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """
    Melakukan normalisasi L2 pada setiap baris matriks embedding.
    Vektor baru v_norm = v / ||v||_2 sehingga ||v_norm||_2 = 1.0.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Hindari pembagian dengan nol jika ada baris ber-norma 0
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def verify_l2_normalization(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Mengembalikan matriks ter-normalisasi dan status verifikasi norm == 1.0."""
    normed_matrix = normalise_embeddings(matrix)
    row_norms = np.linalg.norm(normed_matrix, axis=1)
    all_equal_one = bool(np.allclose(row_norms, 1.0))
    return normed_matrix, all_equal_one


# -------------------------------------------------------------------
# 3. Analisis Folder Teks ke Pandas DataFrame
# -------------------------------------------------------------------
def analyze_txt_folder(folder_path: str | Path) -> pd.DataFrame:
    """
    Membaca seluruh file .txt di folder dan mengembalikan DataFrame
    dengan kolom: filename, char_count, word_count, sentence_count.
    Diurutkan berdasarkan word_count secara descending.
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {folder}")

    records = []
    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        char_count = len(text)
        word_count = len(text.split())
        # Memisah kalimat berdasarkan tanda titik, tanda seru, atau tanda tanya
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = len(sentences)

        records.append({
            "filename": file_path.name,
            "char_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="word_count", ascending=False).reset_index(drop=True)
    return df


# -------------------------------------------------------------------
# 4. Pairwise Cosine Similarity Matrix (Hash-based Mock Embeddings)
# -------------------------------------------------------------------
def get_hash_embedding(text: str, dim: int = 16) -> np.ndarray:
    """Menghasilkan mock embedding deterministik berukuran `dim` berdasarkan hash string."""
    hash_hex = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed = int(hash_hex[:8], 16)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def compute_pairwise_cosine(corpus: list[str]) -> tuple[np.ndarray, tuple[str, str, float]]:
    """
    Menghitung matriks cosine similarity antar seluruh pasangan kalimat dalam corpus.
    Mengembalikan matriks similarity dan pasangan kalimat dengan similarity tertinggi.
    """
    embeddings = np.array([get_hash_embedding(text) for text in corpus])
    normed_embeddings = normalise_embeddings(embeddings)

    # Matriks perkalian titik (Dot Product dari L2-normalized vectors == Cosine Similarity)
    similarity_matrix = normed_embeddings @ normed_embeddings.T

    n = len(corpus)
    max_sim = -1.0
    best_pair = ("", "", -1.0)

    for i in range(n):
        for j in range(i + 1, n):
            sim = float(similarity_matrix[i, j])
            if sim > max_sim:
                max_sim = sim
                best_pair = (corpus[i], corpus[j], round(sim, 4))

    return similarity_matrix, best_pair


if __name__ == "__main__":
    print("=== Exercise 1: LLM Benchmark Analysis ===")
    benchmark_csv = Path("benchmark_scores_test.csv")
    generate_benchmark_csv(benchmark_csv)

    results = analyze_benchmark_scores(benchmark_csv)
    print("1. Rata-rata Score per Model:")
    for model, mean_score in results["mean_score_per_model"].items():
        print(f"   - {model}: {mean_score}")

    print("\n2. Task Terbaik per Model:")
    for item in results["best_task_per_model"]:
        print(f"   - {item['model']}: {item['task']} (Score: {item['score']})")

    print(f"\n3. Korelasi Score & Latency: {results['score_latency_correlation']}")
    benchmark_csv.unlink(missing_ok=True)

    print("\n=== Exercise 2: L2 Normalization ===")
    rng = np.random.default_rng(100)
    raw_embeddings = rng.standard_normal((5, 8))
    normed_embeddings, is_valid = verify_l2_normalization(raw_embeddings)
    row_norms = np.linalg.norm(normed_embeddings, axis=1)

    print("Raw embeddings shape:", raw_embeddings.shape)
    print("Norm tiap baris setelah normalisasi:", np.round(row_norms, 4))
    print("Apakah seluruh norm sama dengan 1.0?:", is_valid)

    print("\n=== Exercise 3: Text Files Analysis DataFrame ===")
    test_folder = Path("temp_txt_samples")
    test_folder.mkdir(exist_ok=True)

    (test_folder / "doc_short.txt").write_text("Generative AI is awesome. Python rules!", encoding="utf-8")
    (test_folder / "doc_medium.txt").write_text("Large Language Models use transformers. They process tokens. They produce text.", encoding="utf-8")
    (test_folder / "doc_long.txt").write_text("Retrieval Augmented Generation combines vector search with LLMs. It reduces hallucinations significantly. This improves factual accuracy in production systems.", encoding="utf-8")

    txt_df = analyze_txt_folder(test_folder)
    print(txt_df.to_string(index=False))

    # Clean up test folder
    for f in test_folder.glob("*.txt"):
        f.unlink()
    test_folder.rmdir()

    print("\n=== Exercise 4: Pairwise Cosine Similarity ===")
    corpus = [
        "What is Retrieval Augmented Generation?",
        "How do transformers work in deep learning?",
        "Explain vector databases for similarity search.",
        "Python core data structures and comprehensions.",
        "How to evaluate LLM output accuracy?",
    ]

    sim_matrix, (s1, s2, highest_sim) = compute_pairwise_cosine(corpus)
    print("Matriks Cosine Similarity (5x5):")
    print(np.round(sim_matrix, 3))

    print("\nPasangan Kalimat dengan Similarity Tertinggi:")
    print(f"  Kalimat 1 : '{s1}'")
    print(f"  Kalimat 2 : '{s2}'")
    print(f"  Similarity: {highest_sim}")