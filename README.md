# Large Language Model-Guided Graph Clustering for Open Research Data Corpora

Undergraduate thesis (skripsi) project — Computer Science, Bina Nusantara University. An end-to-end modular pipeline for clustering and trend analysis of AI research literature, combining scientific language models, graph neural networks, density-based clustering, and LLM-based topic labeling.

## Abstract

Frequency-based text mining techniques often fail to capture deeper semantic context and ignore the structural properties of scholarly communication networks. This project proposes a modular pipeline that uses **SPECTER** to produce dense semantic document vectors, builds a document similarity graph via **k-NN (FAISS)**, applies **GraphSAGE** for neighborhood aggregation, reduces dimensionality with **UMAP**, and clusters the result with **HDBSCAN**. Cluster keywords are extracted via class-based **TF-IDF**, and an **LLM (Llama-3.3-70b via Groq API)** generates human-readable topic labels. Applied to ~94,600 AI conference papers (2000–2025), the pipeline detects research communities and analyzes topic trends over time — showing a sharp rise in generative AI topics (e.g. "AI Generated Image Detection") and a decline in COVID-19-related research.

## Pipeline

The pipeline consists of eleven consecutive steps:

1. **Data Loading & Preprocessing** — deduplication, null filtering, HTML/XML stripping, year filtering (2000–2025)
2. **Semantic Encoding** — SPECTER embeddings (768-dim) from concatenated title + summary
3. **Dimensionality Reduction (1st pass)** — UMAP 768 → 64 dimensions
4. **Graph Construction** — FAISS `IndexFlatIP` k-NN graph (k=10, cosine similarity)
5. **Representation Learning** — two-layer GraphSAGE (link prediction w/ negative sampling), 768 → 256 → 128 dimensions
6. **Dimensionality Reduction (2nd pass)** — UMAP 128 → 32 dimensions
7. **Clustering** — HDBSCAN, hyperparameters tuned via Optuna (TPESampler, 50 trials)
8. **Keyword Extraction** — class-based TF-IDF (top 10 terms/cluster)
9. **Topic Labeling** — LLM (Llama-3.3-70b via Groq API) generates label, description, confidence score
10. **Evaluation** — Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
11. **Trend Analysis** — topic ranking, growth-rate analysis (2020 vs. 2024), era-based analysis (2000–2009 / 2010–2019 / 2020–2025)

## Results

- **Dataset:** 94,643 papers (filtered from a raw 179,161-record corpus scraped from arXiv & Semantic Scholar, via Kaggle)
- **Graph:** 94,643 nodes, 946,430 directed edges (mean degree 10.0)
- **Clusters:** 290 topic clusters (noise rate 22.8%)
- **Evaluation metrics** (on 32-dim UMAP-reduced GNN embeddings):
  | Metric | Score |
  |---|---|
  | Silhouette Score | 0.6176 |
  | Davies-Bouldin Index | 0.5018 |
  | Calinski-Harabasz Score | 62,404.81 |
- **Top research topic overall:** AI in Education (4,118 papers)
- **Fastest-growing topics (2020→2024):** AI Generated Image Detection (+2,350%), Clinical NLP and LLMs (+2,120%), AI Generated Text Detection (+1,400%)
- **Fastest-declining topics:** HIV Prevention in MSM (-82%), AI in COVID-19 Management (-76%), Variational Autoencoders (-75%)

## Project Structure

```
.
├── data/               # Raw and processed datasets (see note below)
├── notebooks/          # Pipeline development notebooks
├── src/
│   ├── preprocessing/  # Data cleaning, filtering, text_input construction
│   ├── embedding/      # SPECTER embedding generation
│   ├── graph/          # UMAP + FAISS kNN graph construction
│   ├── gnn/            # GraphSAGE model (representation learning)
│   ├── clustering/     # HDBSCAN + Optuna tuning
│   ├── labeling/       # TF-IDF keyword extraction + LLM topic labeling
│   └── trend_analysis/ # Era-based, growth-rate, and ranking analysis
├── figures/             # Visualizations used in the thesis document
├── requirements.txt
└── README.md
```

> **Note on data:** The full dataset is not included in this repository due to size. Source: ["AI Articles scrapped from arXiv & SemanticScholar"](https://www.kaggle.com/datasets/ddarryl/ai-articles-scrapped-from-arxiv-and-semanticscholar) (Kaggle).

## Setup

```bash
git clone https://github.com/<username>/llm-guided-graph-clustering.git
cd llm-guided-graph-clustering
pip install -r requirements.txt
```

You'll also need a **Groq API key** for the LLM-based topic labeling stage (set as an environment variable, e.g. `GROQ_API_KEY`).

## Usage

```bash
# Example — adjust to your actual scripts/notebooks
python src/preprocessing/clean_and_filter.py
python src/embedding/generate_embeddings.py
python src/graph/build_knn_graph.py
python src/gnn/train_graphsage.py
python src/clustering/run_hdbscan_optuna.py
python src/labeling/extract_keywords_and_label.py
python src/trend_analysis/analyze_trends.py
```

## Tech Stack

- Python
- [SPECTER](https://github.com/allenai/specter) (via SentenceTransformers) — scientific document embeddings
- UMAP — dimensionality reduction
- FAISS — approximate/exact nearest-neighbor graph construction
- PyTorch Geometric (GraphSAGE) — graph representation learning
- HDBSCAN + Optuna — density-based clustering with automated hyperparameter search
- scikit-learn — TF-IDF keyword extraction
- Groq API (Llama-3.3-70b-versatile) — LLM-based topic labeling

## Authors

- Joey Martin
- Jessica Debora
- Samantha Niandra
- Alexander Agung Santoso Gunawan (advisor)

Computer Science Department, School of Computer Science, Bina Nusantara University, Jakarta, Indonesia

## Citation

If you use this pipeline, please cite the corresponding paper (full citation TBD upon publication).

## License

[Pilih lisensi, misal MIT — atau kosongkan dulu kalau belum diputuskan]
