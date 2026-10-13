"""Step 3: run questions.csv through the bot and report answer/refusal behavior.
Edit questions.csv so the 'yes' questions match YOUR documents. Then read results.csv by hand."""
import csv
from rag import answer, REFUSAL

rows, correct = [], 0
with open("questions.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        text, sources = answer(r["question"])
        refused = text.startswith(REFUSAL[:30])
        ok = (not refused) if r["should_answer"] == "yes" else refused
        correct += ok
        rows.append({**r, "answer": text.replace("\n", " "), "sources": ";".join(sources),
                     "refused": refused, "behavior_correct": ok})
with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"Answer/refuse behavior correct: {correct}/{len(rows)}")
print("Now manually check the 'yes' answers in results.csv for factual accuracy.")
