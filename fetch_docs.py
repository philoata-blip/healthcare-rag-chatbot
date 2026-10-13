"""Download ~28 MedlinePlus health-topic summaries (US National Library of Medicine, free web service)
into data/*.txt for the RAG chatbot. Run once: python fetch_docs.py"""
import html, os, re, time
import xml.etree.ElementTree as ET
import requests

TOPICS = [
    "Diabetes", "Type 2 Diabetes", "High Blood Pressure", "Heart Diseases", "Stroke", "Asthma",
    "Anemia", "Dehydration", "Smoking", "Exercise and Physical Fitness", "Sleep Disorders",
    "Obesity", "Cholesterol", "Kidney Diseases", "Liver Diseases", "Influenza", "COVID-19",
    "Pneumonia", "Depression", "Anxiety", "Headache", "Back Pain", "Osteoporosis", "Arthritis",
    "Food Allergies", "Vitamins", "Healthy Eating", "Cancer",
]
URL = "https://wsearch.nlm.nih.gov/ws/query"
os.makedirs("data", exist_ok=True)


def clean(raw):
    text = re.sub(r"<[^>]+>", " ", html.unescape(raw))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def fetch(topic):
    r = requests.get(URL, params={"db": "healthTopics", "term": f'title:"{topic}"', "retmax": 5,
                                  "tool": "student-rag-demo", "email": "student@example.com"}, timeout=30)
    r.raise_for_status()
    best = None
    for doc in ET.fromstring(r.content).iter("document"):
        fields = {c.get("name"): "".join(c.itertext()) for c in doc.findall("content")}
        title = clean(fields.get("title", ""))
        summary = clean(fields.get("FullSummary", ""))
        if not summary:
            continue
        item = (title, summary, doc.get("url", ""))
        if title.lower() == topic.lower():
            return item
        best = best or item
    return best


saved = 0
for t in TOPICS:
    try:
        item = fetch(t)
    except Exception as e:
        print("FAILED", t, e)
        continue
    if not item:
        print("not found", t)
        continue
    title, summary, url = item
    name = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") + ".txt"
    with open(os.path.join("data", name), "w", encoding="utf-8") as f:
        f.write(f"{title}\nSource: MedlinePlus.gov, U.S. National Library of Medicine. {url}\n\n{summary}\n")
    saved += 1
    print("saved", name, len(summary.split()), "words")
    time.sleep(1)
print(f"{saved} files in data/")
