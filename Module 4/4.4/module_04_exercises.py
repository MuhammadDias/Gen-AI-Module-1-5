# 4.4 Module 04 Exercises
import asyncio
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
import httpx

# -------------------------------------------------------------------
# 1. Serialisasi & Deserialisasi Percakapan (JSON)
# -------------------------------------------------------------------
def save_conversation(history: list[dict], path: str | Path) -> None:
    """Menyimpan daftar pesan percakapan ke file JSON menggunakan pathlib."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def load_conversation(path: str | Path) -> list[dict]:
    """Membaca dan mengembalikan daftar pesan percakapan dari file JSON."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File percakapan tidak ditemukan: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


# -------------------------------------------------------------------
# 2. Async Endpoint Comparison dengan HTTPX
# -------------------------------------------------------------------
async def _fetch_single_endpoint(client: httpx.AsyncClient, url: str) -> tuple[str, int, float]:
    """Helper untuk mengukur latensi dan status code dari 1 URL."""
    start_time = time.perf_counter()
    try:
        response = await client.get(url, timeout=5.0)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return (url, response.status_code, round(elapsed_ms, 2))
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return (url, 0, round(elapsed_ms, 2))


async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    """Memanggil beberapa URL API secara konkuren menggunakan httpx."""
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_single_endpoint(client, url) for url in urls]
        results = await asyncio.gather(*tasks)
        return list(results)


# -------------------------------------------------------------------
# 3. Config Loader dengan Override Environment Variables
# -------------------------------------------------------------------
def load_config(config_path: str | Path, env_prefix: str = "") -> dict:
    """
    Membaca file konfigurasi JSON dan menggabungkannya dengan nilai dari environment variable.
    Nilai dari environment variable memiliki prioritas lebih tinggi (override).
    """
    file_path = Path(config_path)
    config: dict = {}
    if file_path.exists():
        config = json.loads(file_path.read_text(encoding="utf-8"))

    merged_config = dict(config)
    for key in config.keys():
        env_var_name = f"{env_prefix}{key.upper()}"
        if env_var_name in os.environ:
            val = os.environ[env_var_name]
            # Tipe data casting sederhana (bool, int, float, str)
            if val.lower() == "true":
                merged_config[key] = True
            elif val.lower() == "false":
                merged_config[key] = False
            else:
                try:
                    merged_config[key] = int(val)
                except ValueError:
                    try:
                        merged_config[key] = float(val)
                    except ValueError:
                        merged_config[key] = val

    return merged_config


# -------------------------------------------------------------------
# 4. Thread-Safe CSV Log Writer
# -------------------------------------------------------------------
class ThreadSafeCSVLogger:
    """Class untuk mencatat log pemanggilan LLM ke file CSV secara thread-safe."""
    def __init__(self, filename: str | Path):
        self.filepath = Path(filename)
        self.lock = threading.Lock()

        # Inisialisasi header jika file belum dibuat
        with self.lock:
            if not self.filepath.exists():
                self.filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "latency_ms"])

    def log(self, model: str, input_tokens: int, output_tokens: int, latency_ms: float) -> None:
        timestamp = datetime.now().isoformat()
        with self.lock:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, model, input_tokens, output_tokens, round(latency_ms, 2)])


if __name__ == "__main__":
    print("=== Exercise 1: JSON Conversation Storage ===")
    sample_history = [
        {"role": "user", "content": "Apa bedanya RAG dan Fine-tuning?"},
        {"role": "assistant", "content": "RAG mengambil dokumen eksternal secara dinamis, sedangkan fine-tuning mengubah bobot model."},
    ]
    json_file = Path("conversation_test.json")
    save_conversation(sample_history, json_file)
    print(f"Disimpan ke '{json_file.name}'")

    loaded_history = load_conversation(json_file)
    print("Data terload:", loaded_history)
    json_file.unlink(missing_ok=True)  # Hapus file temporary test

    print("\n=== Exercise 2: Async Compare Endpoints ===")
    test_urls = [
        "https://httpbin.org/status/200",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/404",
    ]
    print(f"Memeriksa {len(test_urls)} endpoint secara simultan...")
    endpoint_results = asyncio.run(compare_endpoints(test_urls))
    for url, status, latency in endpoint_results:
        print(f"URL: {url} -> Status: {status}, Latency: {latency}ms")

    print("\n=== Exercise 3: Config Loader + Env Override ===")
    config_file = Path("app_config_test.json")
    default_config = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    config_file.write_text(json.dumps(default_config, indent=2), encoding="utf-8")

    # Set environment variable override
    os.environ["MODEL"] = "gpt-4o"
    os.environ["TEMPERATURE"] = "0.2"

    merged = load_config(config_file)
    print("File Config Asli:", default_config)
    print("ENV Overrides (MODEL=gpt-4o, TEMPERATURE=0.2):")
    print("Config Tergabung:", merged)
    config_file.unlink(missing_ok=True)

    print("\n=== Exercise 4: Thread-Safe CSV Logger ===")
    csv_file = Path("llm_calls_log.csv")
    logger = ThreadSafeCSVLogger(csv_file)

    def worker_task(thread_id: int):
        for j in range(2):
            logger.log(
                model=f"model-v{thread_id}",
                input_tokens=100 + thread_id * 10 + j,
                output_tokens=50 + j * 5,
                latency_ms=120.5 + thread_id * 15,
            )

    threads = [threading.Thread(target=worker_task, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"Isi file CSV '{csv_file.name}':")
    print(csv_file.read_text(encoding="utf-8"))
    csv_file.unlink(missing_ok=True)