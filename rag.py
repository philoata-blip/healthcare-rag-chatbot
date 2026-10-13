"""Core RAG logic: retrieve relevant chunks, then ask the LLM to answer only from them."""
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

_embedder = SentenceTransformer(config.EMBED_MODEL)
_index = faiss.read_index(config.INDEX_FILE)
with open(config.CHUNKS_FILE, encoding="utf-8") as f:
    _chunks = json.load(f)

_tok = AutoTokenizer.from_pretrained(config.LLM_MODEL)
_llm = AutoModelForCausalLM.from_pretrained(config.LLM_MODEL, torch_dtype="auto")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_llm.to(DEVICE)
_llm.eval()

REFUSAL = "I don\u0027t have enough information in my sources to answer that."

SYSTEM_PROMPT = (
    "You are a careful healthcare information assistant. Answer ONLY using the provided context. "
    "If the context does not contain the answer, reply exactly: " + REFUSAL + " "
    "Do not diagnose, prescribe, or give dosages. Keep the answer under 120 words."
)


def retrieve(question, k=config.TOP_K):
    q = _embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = _index.search(q, k)
    return [(float(s), _chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]


def answer(question):
    hits = retrieve(question)
    if not hits or hits[0][0] < config.MIN_SCORE:
        return REFUSAL, []
    context = "\n\n".join(f"[{h['source']}] {h['text']}" for _, h in hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    prompt = _tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = _llm.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
    text = _tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    sources = sorted({h["source"] for _, h in hits})
    return text, sources
