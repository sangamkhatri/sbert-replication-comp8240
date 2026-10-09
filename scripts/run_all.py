import time

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, util

print("Loading SBERT (sentence-transformers/bert-base-nli-mean-tokens) ...")
t0 = time.time()
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
print(f"Loaded in {time.time() - t0:.1f}s")

def run_eval(name, dataset_ids, split):
    print(f"\n--- {name} ---")
    ds = None
    for dataset_id in dataset_ids:
        try:
            ds = load_dataset(dataset_id, split=split)
            print(f"Loaded '{dataset_id}' ({len(ds)} pairs)")
            break
        except Exception as e:
            print(f"Could not load '{dataset_id}': {e}")
    if ds is None:
        print(f"SKIPPED {name}: could not load any dataset id")
        return None

    s1, s2 = ds["sentence1"], ds["sentence2"]
    gold = np.array(ds["score"], dtype=float)
    emb1 = model.encode(s1, convert_to_tensor=True, show_progress_bar=True)
    emb2 = model.encode(s2, convert_to_tensor=True, show_progress_bar=True)
    cos = util.cos_sim(emb1, emb2).cpu().numpy().diagonal()
    rho, _ = spearmanr(cos, gold)
    print(f"Spearman correlation on {name}: {rho * 100:.2f}")
    return rho * 100

sts_rho = run_eval(
    "STS Benchmark (test)",
    ["sentence-transformers/stsb", "mteb/stsbenchmark-sts"],
    "test",
)

biosses_rho = run_eval(
    "BIOSSES",
    ["mteb/biosses-sts"],
    "test",
)

print("\n" + "=" * 60)
print("FINAL RESULTS -- copy these two lines back to Claude:")
print(f"STS Benchmark Spearman: {sts_rho:.2f}  (paper reports 77.03)" if sts_rho else "STS Benchmark: FAILED")
print(f"BIOSSES Spearman: {biosses_rho:.2f}  (not in original paper)" if biosses_rho else "BIOSSES: FAILED")
print("=" * 60)
