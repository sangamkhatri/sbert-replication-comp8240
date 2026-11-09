"""
COMP8240 Final Report -- adds SBERT's own cosine similarity to the 8
headline pairs (human + LLM-judge scores already filled in by
results/headlines_annotated.csv) and computes the two correlations that
go in "Results on new data":
  1. SBERT cosine sim vs. human score      (the real evaluation result,
     same Spearman-rho framing as STS-B/BIOSSES)
  2. SBERT cosine sim vs. LLM-judge score  (lets the report say whether the
     LLM-judge agrees with SBERT about as well as the human annotator does)

Run this in the Codespace (needs sentence-transformers), after
results/headlines_annotated.csv exists (already committed to the repo).

Usage:
    python scripts/score_headlines_sbert.py
Output:
    results/headlines_final.csv
    results/headlines_sbert_vs_human.txt
"""

import csv

from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util

rows = []
with open("results/headlines_annotated.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} annotated headline pairs. Loading SBERT ...")
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

sbert_scores = []
for row in rows:
    emb_a, emb_b = model.encode([row["headline_a"], row["headline_b"]], convert_to_tensor=True)
    sim = util.cos_sim(emb_a, emb_b).item()
    sbert_scores.append(sim)
    row["sbert_cosine_sim"] = f"{sim:.4f}"

human = [float(r["human_score_0_to_5"]) for r in rows]
judge = [float(r["llm_judge_score_0_to_5"]) for r in rows]

rho_human, p_human = spearmanr(sbert_scores, human)
rho_judge, p_judge = spearmanr(sbert_scores, judge)

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print(f"SBERT cosine sim vs. human score:     Spearman rho = {rho_human:.3f} (p={p_human:.4f})")
print(f"SBERT cosine sim vs. LLM-judge score: Spearman rho = {rho_judge:.3f} (p={p_judge:.4f})")
for row in rows:
    print(f"  cos_sim={row['sbert_cosine_sim']} | human={row['human_score_0_to_5']} | "
          f"judge={row['llm_judge_score_0_to_5']} | {row['headline_a'][:40]}... / {row['headline_b'][:40]}...")
print("=" * 60)

with open("results/headlines_final.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = ["headline_a", "headline_b", "human_score_0_to_5", "llm_judge_score_0_to_5",
                  "sbert_cosine_sim", "rationale"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for row in rows:
        w.writerow({k: row.get(k, "") for k in fieldnames})

with open("results/headlines_sbert_vs_human.txt", "w") as f:
    f.write(f"SBERT cosine sim vs. human score:     Spearman rho = {rho_human:.3f} (p={p_human:.4f})\n")
    f.write(f"SBERT cosine sim vs. LLM-judge score: Spearman rho = {rho_judge:.3f} (p={p_judge:.4f})\n")

print("\nWrote results/headlines_final.csv and results/headlines_sbert_vs_human.txt")