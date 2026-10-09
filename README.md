# SBERT Replication — COMP8240 Applications of Data Science

Replicability study of **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**
(Reimers & Gurevych, EMNLP-IJCNLP 2019).

- Paper: [ACL Anthology (D19-1410)](https://aclanthology.org/D19-1410) · [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- Official code: [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- Author: Sangam Khatri (Student ID 49092502), Master of Data Science, Macquarie University

## Project overview

SBERT produces fixed-length sentence embeddings using a siamese BERT network with mean
pooling, so that semantically similar sentences end up close together under cosine
similarity. This repo reproduces the paper's core result, evaluates the released
checkpoint on a dataset the paper never tested, and begins building a new dataset of
Australian news headlines.

## Repository structure
├── requirements.txt
├── scripts/
│ ├── run_all.py # Reproduction: STS Benchmark + BIOSSES
│ └── build_headlines_dataset.py # New dataset: headline pairing + SBERT preview
└── results/
└── headlines_pairs_to_annotate.csv

## Replication status

| Task | Status | Result |
|---|---|---|
| Feasibility check | Done | Confirmed SBERT loads and scores paraphrase/unrelated sentence pairs sensibly |
| STS Benchmark reproduction | Done | Spearman **76.98** (paper reports 77.03 for this checkpoint) |
| Existing-dataset evaluation (BIOSSES) | Done | Spearman **51.62** — notably lower, suggesting limited transfer to biomedical text |
| New dataset (Australian headlines) | In progress | 12 real headlines collected from abc.net.au/news, paired (8 pairs), SBERT preview scores generated |
| Remaining existing datasets (PAWS, Quora QP, Twitter PIT, CQADupStack) | Planned | Scheduled for the final report |

## Model

`sentence-transformers/bert-base-nli-mean-tokens` — the original released checkpoint
from the paper, loaded via the `sentence-transformers` library.

## How to reproduce

```bash
pip install -r requirements.txt
python scripts/run_all.py          # STS Benchmark + BIOSSES
python scripts/build_headlines_dataset.py   # new-data seed
```

Both scripts run on CPU; no GPU required. `run_all.py` prints a `FINAL RESULTS` block
with Spearman correlations for STS Benchmark and BIOSSES. `build_headlines_dataset.py`
writes `results/headlines_pairs_to_annotate.csv` and prints SBERT's cosine similarity
for each headline pair.

## New data

The proposal plans five extensions beyond the original paper: four existing datasets
not used by the authors (PAWS, Quora Question Pairs, the Twitter Paraphrase Corpus,
CQADupStack) and one newly constructed dataset — an STS-style set of Australian news
headlines, eventually built from the full ["A Million Headlines"](https://www.kaggle.com/datasets/therohk/million-headlines)
ABC News collection on Kaggle (2003–present). `build_headlines_dataset.py` is the seed
for that dataset: it pairs a small batch of real, current headlines and generates an
annotation sheet; the next step is annotating a full 100–150 pairs and scaling up
collection from the full corpus.
