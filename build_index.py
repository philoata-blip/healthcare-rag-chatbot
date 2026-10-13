"""Step 1: read data/*.txt, split into overlapping chunks, embed, save a FAISS index."""
import glob, json, os
import faiss
from sentence_transformers import SentenceTransformer
import config


def chunk_words(text, size, overlap):
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def main():
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.txt")))
    if not files:
        raise SystemExit("No .txt files found in data/. Add 20-30 health fact sheets first.")
    chunks = []
    for path in files:
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for c in chunk_words(text, config.CHUNK_WORDS, config.CHUNK_OVERLAP):
            chunks.append({"source": source, "text": c})
    print(f"{len(files)} files -> {len(chunks)} chunks")

    model = SentenceTransformer(config.EMBED_MODEL)
    emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True,
                       show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])   # inner product == cosine on normalized vectors
    index.add(emb)
    faiss.write_index(index, config.INDEX_FILE)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print("Saved", config.INDEX_FILE, "and", config.CHUNKS_FILE)


if __name__ == "__main__":
    main()
