"""
COMP8240 Final Report -- "one more existing dataset" extension.

Run this in the GitHub Codespace (same environment that produced the
STS Benchmark / BIOSSES numbers in results/run_all_output.txt). It needs
sentence-transformers + datasets, same as scripts/run_all.py.

WHY QUORA QUESTION PAIRS, AND WHY THE METRIC IS DIFFERENT:
Quora QP (and PAWS) do NOT have continuous 0-5 human similarity scores like
STS-B/BIOSSES -- they have a BINARY label (1 = duplicate/paraphrase,
0 = not). So Spearman-correlation-against-gold-score (the metric used
everywhere else in this project) doesn't directly apply. Instead we report
the three standard things people report for a binary-label + continuous-score
setup, so the new result is still comparable to the rest of the project:

  1. Mean (and SD) of SBERT cosine similarity, split by gold label.
     -> directly answers "does the model's similarity score actually track
        the human duplicate/not-duplicate judgement?"
  2. AUROC of cosine similarity as a duplicate-detector.
     -> standard way to summarise separation without picking a threshold.
  3. Point-biserial correlation between the binary label and the cosine
     score -- this is mathematically identical to a Pearson correlation
     and is the natural "Spearman-style" analogue for a binary gold label,
     so it lets the report say "correlation between SBERT similarity and
     human judgement" in the same spirit as the other datasets.

Usage:
    python run_quora_extension.py
Output:
    results/quora_extension_output.txt  (also printed to stdout)
    results/quora_extension_pairs.csv   (sampled pairs + scores, for the
                                          appendix / "quality of new data")
"""

import csv
import time

import numpy as np
from datasets import load_dataset
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
from sentence_transformers import SentenceTransformer, util

N_PER_CLASS = 150  # 150 duplicate + 150 non-duplicate = 300 pairs total
SEED = 42

print("Loading SBERT (sentence-transformers/bert-base-nli-mean-tokens) ...")
t0 = time.time()
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
print(f"Loaded in {time.time() - t0:.1f}s")

print("\nLoading Quora Question Pairs (validation split) ...")
# quora_duplicates mirror on HF; falls back to the original 'quora' id if needed.
ds = None
for dataset_id, split in [
    ("SetFit/qqp", "validation"),
    ("glue", "validation"),  # glue/qqp config
]:
    try:
        if dataset_id == "glue":
            ds = load_dataset("glue", "qqp", split="validation")
        else:
            ds = load_dataset(dataset_id, split=split)
        print(f"Loaded '{dataset_id}' ({len(ds)} pairs)")
        break
    except Exception as e:
        print(f"Could not load '{dataset_id}': {e}")

if ds is None:
    raise SystemExit("Could not load Quora QP from any known HF dataset id.")

# Normalise column names across the two possible sources.
cols = ds.column_names
q1_col = "text1" if "text1" in cols else ("question1" if "question1" in cols else "sentence1")
q2_col = "text2" if "text2" in cols else ("question2" if "question2" in cols else "sentence2")
label_col = "label" if "label" in cols else "is_duplicate"

labels = np.array(ds[label_col])
dup_idx = np.where(labels == 1)[0]
nondup_idx = np.where(labels == 0)[0]

rng = np.random.default_rng(SEED)
sample_idx = np.concatenate([
    rng.choice(dup_idx, size=min(N_PER_CLASS, len(dup_idx)), replace=False),
    rng.choice(nondup_idx, size=min(N_PER_CLASS, len(nondup_idx)), replace=False),
])
rng.shuffle(sample_idx)

q1 = [ds[int(i)][q1_col] for i in sample_idx]
q2 = [ds[int(i)][q2_col] for i in sample_idx]
gold = np.array([ds[int(i)][label_col] for i in sample_idx], dtype=int)

print(f"\nEncoding {len(sample_idx)} sampled pairs ({N_PER_CLASS} per class) ...")
emb1 = model.encode(q1, convert_to_tensor=True, show_progress_bar=True)
emb2 = model.encode(q2, convert_to_tensor=True, show_progress_bar=True)
cos = util.cos_sim(emb1, emb2).cpu().numpy().diagonal()

mean_dup, sd_dup = cos[gold == 1].mean(), cos[gold == 1].std()
mean_nondup, sd_nondup = cos[gold == 0].mean(), cos[gold == 0].std()
auroc = roc_auc_score(gold, cos)
r_pb, p_pb = pointbiserialr(gold, cos)

print("\n" + "=" * 60)
print("FINAL RESULTS:")
print(f"N pairs: {len(sample_idx)} ({N_PER_CLASS} duplicate / {N_PER_CLASS} non-duplicate)")
print(f"Mean cosine sim, duplicate pairs:     {mean_dup:.3f} (sd={sd_dup:.3f})")
print(f"Mean cosine sim, non-duplicate pairs: {mean_nondup:.3f} (sd={sd_nondup:.3f})")
print(f"AUROC (cosine sim as duplicate-detector): {auroc:.3f}")
print(f"Point-biserial correlation (label vs. cosine sim): {r_pb:.3f} (p={p_pb:.1e})")
print("=" * 60)

with open("results/quora_extension_output.txt", "w") as f:
    f.write(f"N pairs: {len(sample_idx)} ({N_PER_CLASS} duplicate / {N_PER_CLASS} non-duplicate)\n")
    f.write(f"Mean cosine sim, duplicate pairs:     {mean_dup:.3f} (sd={sd_dup:.3f})\n")
    f.write(f"Mean cosine sim, non-duplicate pairs: {mean_nondup:.3f} (sd={sd_nondup:.3f})\n")
    f.write(f"AUROC (cosine sim as duplicate-detector): {auroc:.3f}\n")
    f.write(f"Point-biserial correlation (label vs. cosine sim): {r_pb:.3f} (p={p_pb:.1e})\n")

with open("results/quora_extension_pairs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question1", "question2", "gold_label", "sbert_cosine_sim"])
    for a, b, g, c in zip(q1, q2, gold, cos):
        writer.writerow([a, b, int(g), f"{c:.4f}"])

print("\nWrote results/quora_extension_output.txt and results/quora_extension_pairs.csv")