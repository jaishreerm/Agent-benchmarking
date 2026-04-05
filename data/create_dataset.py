from datasets import load_dataset
import json

print("🚀 Starting...")

try:
    dataset = load_dataset("squad", split="train[:1000]")
    print("✅ Dataset loaded!")
except Exception as e:
    print("❌ Error loading dataset:", e)

data = []

for item in dataset:
    data.append({
        "question": item["question"],
        "context": item["context"],
        "answer": item["answers"]["text"][0]
    })

print(f"✅ Processed {len(data)} items")

try:
    with open("data/dataset.json", "w") as f:
        json.dump(data, f, indent=2)
    print("✅ Dataset saved!")
except Exception as e:
    print("❌ Error saving file:", e)