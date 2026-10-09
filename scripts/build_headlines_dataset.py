"""
COMP8240 Project Update - progress on the new Australian-headlines dataset.

Quick-start version: real, CURRENT Australian news headlines (copied from
abc.net.au/news) below. This script will:
1. Pair them up -- some deliberately from the same topic (likely similar),
   some at random (likely dissimilar), following the plan in the proposal.
2. Write out a CSV with blank 'human_score' columns for you to fill in by
   hand (0-5 scale, same convention as the STS Benchmark).
3. Print a couple of pairs scored by SBERT right away, so you have a first
   look at how the model handles headline-style text.

This is a SEED for the full dataset (eventually built from the "A Million
Headlines" ABC News collection on Kaggle, as described in the proposal) --
it's enough to show real collection + pairing + annotation progress today.
"""

import csv
import itertools
import random

from sentence_transformers import SentenceTransformer, util

# --- 1. COLLECTION ---------------------------------------------------------
# Collected live from abc.net.au/news on 2026-10-09.
HEADLINES = [
    "ATO delays credit card ban as government backs down after business backlash",
    "How the card surcharge ban could change our relationship with cash",
    "Australian tail exposed as batting woes continues in South Africa",
    "Litchfield fireworks help Australia to ODI victory over Bangladesh",
    "Families of men killed on Mexico surf trip call for maximum sentences",
    "Person of interest given alibi in schoolgirl's disappearance",
    "Sister of 13yo critically injured in alleged arson pleads with other teens",
    "Influencer Sam Wood to lodge Supreme Court bail application on DV charges",
    "Two stolen Renoir paintings found, six arrested",
    "Fiona O'Byrne wins 2026 Lester Prize with painting of daughter",
    "Bates dedicates historic Bathurst lap to late mum after stunning qualifying",
    "Woman, 71, treated on 'cold floor' of Adelaide hospital",
]

# --- 2. PAIRING --------------------------------------------------------
# Deliberately-similar pairs: consecutive headlines above are written to be
# about the same topic (indices 0&1, 2&3, 4&5, 6&7 are same-topic pairs).
similar_pairs = [(HEADLINES[i], HEADLINES[i + 1]) for i in range(0, 8, 2)]

# Dissimilar pairs: random cross-topic combinations.
random.seed(42)
all_pairs = list(itertools.combinations(range(len(HEADLINES)), 2))
same_topic_idx = {(i, i + 1) for i in range(0, 8, 2)}
candidate_dissimilar = [p for p in all_pairs if p not in same_topic_idx]
dissimilar_pairs = [
    (HEADLINES[i], HEADLINES[j]) for i, j in random.sample(candidate_dissimilar, 4)
]

all_annotated_pairs = similar_pairs + dissimilar_pairs

# --- 3. WRITE ANNOTATION SHEET -----------------------------------------
out_path = "results/headlines_pairs_to_annotate.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["headline_a", "headline_b", "human_score_0_to_5"])
    for a, b in all_annotated_pairs:
        writer.writerow([a, b, ""])
print(f"Wrote {len(all_annotated_pairs)} pairs to {out_path} -- fill in human_score_0_to_5 by hand.")

# --- 4. FIRST LOOK WITH SBERT -------------------------------------------
print("\nLoading SBERT to preview how it scores these headline pairs ...")
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

print("\nSBERT cosine similarities (preview only -- human scores still needed):")
for a, b in all_annotated_pairs:
    emb_a, emb_b = model.encode([a, b], convert_to_tensor=True)
    sim = util.cos_sim(emb_a, emb_b).item()
    print(f"  cos_sim={sim:.3f} | A: {a!r}  B: {b!r}")
