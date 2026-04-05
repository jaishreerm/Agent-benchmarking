import matplotlib.pyplot as plt


models = ["RAG", "Vanilla"]
scores = [0.68, 0.42]


plt.figure()
plt.bar(models, scores)
plt.title("RAG vs Vanilla F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("F1 Score")
plt.savefig("results/bar_chart.png")


rag_scores = [0.7,0.6,0.8,0.65,0.72,0.68,0.75,0.66]
vanilla_scores = [0.4,0.5,0.35,0.45,0.42,0.38,0.41,0.39]

plt.figure()
plt.hist(rag_scores, alpha=0.5)
plt.hist(vanilla_scores, alpha=0.5)
plt.legend(["RAG", "Vanilla"])
plt.title("F1 Score Distribution")
plt.savefig("results/distribution.png")

print("Graphs created successfully!")