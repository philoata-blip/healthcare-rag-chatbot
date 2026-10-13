# Healthcare FAQ Chatbot (RAG)

Answers health questions **only from a small set of public documents**, cites the source files,
and refuses when the documents don't contain the answer.

## How it works
1. `build_index.py` splits documents into overlapping chunks and embeds them (MiniLM) into a FAISS index.
2. `rag.py` embeds the question, retrieves the top 3 chunks, and refuses if similarity is below a threshold.
3. Otherwise a small open LLM (Qwen2.5-1.5B-Instruct) answers using only that context.
4. `app.py` serves it in a Gradio chat UI.

## Run
```bash
pip install -r requirements.txt
# put 20-30 .txt files in data/ (WHO / MedlinePlus fact sheets, one topic per file)
python build_index.py
python app.py
python evaluate.py      # answer/refuse behavior on questions.csv
```

## Results
(Fill in after running evaluate.py: behavior accuracy, example good answers, example failures.)

## Limitations
Small model, small corpus, not medical advice. Retrieval quality depends on the documents and chunk size.
