import json
from tqdm import tqdm
from retriever import build_retriever
from rag_pipeline import rag_answer
from vanilla_pipeline import vanilla_answer
from evaluation.evaluate import compute_f1

print("Loading dataset...")
data = json.load(open("data/dataset.json"))

print("Building retriever...")
retriever = build_retriever(data)

results = []

print("Running agents...")

for item in tqdm(data[:50]):   
    q = item["question"]
    truth = item["answer"]

    rag_pred = rag_answer(q, retriever)
    vanilla_pred = vanilla_answer(q)

    rag_f1 = compute_f1(rag_pred, truth)
    vanilla_f1 = compute_f1(vanilla_pred, truth)

    results.append({
        "question": q,
        "rag_pred": rag_pred,
        "vanilla_pred": vanilla_pred,
        "rag_f1": rag_f1,
        "vanilla_f1": vanilla_f1
    })

with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done! Results saved.")

import pandas as pd

df = pd.DataFrame(results)

rag_avg = df["rag_f1"].mean()
vanilla_avg = df["vanilla_f1"].mean()

print("\nFINAL RESULTS:")
print("RAG Avg F1:", rag_avg)
print("Vanilla Avg F1:", vanilla_avg)


with open("results/summary.txt", "w") as f:
    f.write(f"RAG Avg F1: {rag_avg}\n")
    f.write(f"Vanilla Avg F1: {vanilla_avg}\n")