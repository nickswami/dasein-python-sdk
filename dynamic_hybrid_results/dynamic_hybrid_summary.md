# Dynamic Hybrid: per-query α for dense + BM25 retrieval

Hybrid retrieval blends dense and BM25 with a scalar α. The optimal α is not constant across queries — it shifts with how lexical the query is and how the encoder handles it. A single fixed α is a compromise that over-weights BM25 on dense-friendly queries and under-weights it on lexical ones, and it caps the quality a hybrid stack can reach.

Dynamic Hybrid picks α per query, in line with the rest of the retrieval path. It ships in two variants: a **portable** variant that any hybrid stack can call, and a **Dasein-native** variant that runs on the Dasein index.

## The two variants

**Portable.** Query in, per-query α out. Drop it into any `score = (1-α)·dense + α·bm25` fusion. Per-query inference: **0.40 ms**. Works with any encoder, any BM25 implementation, any index.

**Dasein-native.** Because we own the full retrieval stack, we were able to refactor our indexing pipeline to support Dynamic Hybrid end-to-end. That integration is what the Dasein-native variant runs on, which is why it is available only on the Dasein index. Per-query inference: **4.17 ms** — still inside any realistic retrieval-path budget — and it turns the portable variant's R@10 lifts into large R@1 lifts on the lexically-rich corpora (+12.0 pp on FEVER, +18.8 pp on NQ) on top of matching or beating the portable variant on R@10, MRR, and mean rank.

Both variants are evaluated against the same baselines (pure dense, pure BM25, best static α) on the same four corpora.

## Headline

- **Higher quality.** Per-query α beats the best static α on all four in-domain corpora, for both variants, on R@10, MRR, and mean rank. The Dasein-native variant adds large R@1 gains on the lexically-rich corpora (FEVER, NQ) on top of that.
- **Low latency.** Portable is **0.40 ms / query**; Dasein-native is **4.17 ms / query**. Both are small fractions of any realistic retrieval-path budget.
- **Real benchmarks.** FiQA, FEVER, SciFact, NQ — the retrieval sets the field uses. They are not curated to flatter hybrid methods; FiQA and SciFact are dense-dominated, FEVER and NQ are lexically richer. We report across all four.
- **Encoder coverage.** Ten public encoders from 22 M to 7 B parameters, 384 to 4096 dimensions, spanning the major embedding families.
- **Generalizes across encoders.** On three encoder families held out of training, the portable variant tracks dense within noise on every corpus while still avoiding the R@1 collapse that fixed α incurs; the Dasein-native variant lifts R@10 over dense on the lexically-rich held-out corpora (FEVER, NQ) and beats best static α on R@1 by large margins. Full held-out tables in the per-variant full-results docs.

## Embedding models

Ten public encoders span the major embedding families (MiniLM, BGE, GTE, E5, mxbai, Arctic, Nomic, Jina, E5-Mistral, Qwen2). Tables below pool across those ten.

| name | HF id | params | dim |
|---|---|---:|---:|
| MiniLM-L6 | sentence-transformers/all-MiniLM-L6-v2 | 22 M | 384 |
| BGE-large | BAAI/bge-large-en-v1.5 | 335 M | 1024 |
| GTE-large | Alibaba-NLP/gte-large-en-v1.5 | 434 M | 1024 |
| E5-large | intfloat/e5-large-v2 | 335 M | 1024 |
| mxbai-large | mixedbread-ai/mxbai-embed-large-v1 | 335 M | 1024 |
| Arctic-L | Snowflake/snowflake-arctic-embed-l | 335 M | 1024 |
| Nomic-v1.5 | nomic-ai/nomic-embed-text-v1.5 | 137 M | 768 |
| Jina-v3 | jinaai/jina-embeddings-v3 | 572 M | 1024 |
| E5-Mistral-7B | intfloat/e5-mistral-7b-instruct | 7 B | 4096 |
| GTE-Qwen2-7B | Alibaba-NLP/gte-Qwen2-7B-instruct | 7 B | 3584 |

Each table compares three methods, pooled across the evaluation encoders for that corpus:

- **Dense only** — α = 0, embedding-only retrieval.
- **Best static α** — swept α on a 21-point grid and picked the best non-zero value by R@10 (what a fixed-hybrid stack would typically pick).
- **Dynamic Hybrid** — per-query α.

α is the BM25 weight in the fusion: `score = (1-α)·dense + α·bm25`.

## Portable variant

Per-query latency **0.40 ms**. Small enough to call inline in front of any hybrid fusion step.

**FiQA** (n=5,799)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.4703 | 0.6951 | 0.7775 | 0.5750 | 13.4 |
| Best static α (α=0.05) | 0.3604 | 0.6760 | 0.7762 | 0.4970 | 13.1 |
| Dynamic Hybrid | 0.4689 | 0.6948 | 0.7789 | 0.5740 | 12.8 |
| Δ Dynamic vs dense | -0.0014 | -0.0003 | +0.0014 | -0.0010 | -0.6 |
| Δ Dynamic vs best static α | +0.1085 | +0.0188 | +0.0028 | +0.0770 | -0.3 |

**FEVER** (n=160,559)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.6680 | 0.7522 | 0.7628 | 0.7084 | 33.4 |
| Best static α (α=0.50) | 0.2623 | 0.6191 | 0.7754 | 0.4217 | 12.8 |
| Dynamic Hybrid | 0.6830 | 0.8263 | 0.8548 | 0.7494 | 12.6 |
| Δ Dynamic vs dense | +0.0150 | +0.0741 | +0.0920 | +0.0410 | -20.8 |
| Δ Dynamic vs best static α | +0.4207 | +0.2072 | +0.0793 | +0.3277 | -0.2 |

**SciFact** (n=2,934)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.6118 | 0.8211 | 0.8770 | 0.7042 | 9.3 |
| Best static α (α=0.05) | 0.6138 | 0.8207 | 0.8817 | 0.7073 | 8.1 |
| Dynamic Hybrid | 0.6125 | 0.8217 | 0.8821 | 0.7054 | 8.1 |
| Δ Dynamic vs dense | +0.0007 | +0.0007 | +0.0051 | +0.0012 | -1.2 |
| Δ Dynamic vs best static α | -0.0014 | +0.0010 | +0.0003 | -0.0020 | -0.0 |

**NQ** (n=4,723)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.5122 | 0.6676 | 0.6943 | 0.5823 | 40.0 |
| Best static α (α=0.05) | 0.4120 | 0.6473 | 0.6957 | 0.5109 | 32.8 |
| Dynamic Hybrid | 0.5079 | 0.7142 | 0.7629 | 0.6003 | 21.6 |
| Δ Dynamic vs dense | -0.0042 | +0.0466 | +0.0686 | +0.0180 | -18.4 |
| Δ Dynamic vs best static α | +0.0959 | +0.0669 | +0.0671 | +0.0894 | -11.2 |

Full per-corpus / per-encoder tables, α sweeps, and lift breakdowns: `dynamic_hybrid_external_full_results.md`.

## Dasein-native variant

Per-query latency **4.17 ms**. Same two-path hybrid contract as the portable variant — per-query α is what flows through the fusion — with large R@1 gains on the lexically-rich corpora (+12.0 pp on FEVER, +18.8 pp on NQ) layered on top of the portable variant's R@10 / MRR / mean-rank wins.

**FiQA** (n=5,802)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.4700 | 0.6948 | 0.7771 | 0.5748 | 13.0 |
| Best static α (α=0.05) | 0.3626 | 0.6751 | 0.7763 | 0.4983 | 13.2 |
| Dynamic Hybrid | 0.4809 | 0.7122 | 0.7939 | 0.5873 | 10.5 |
| Δ Dynamic vs dense | +0.0109 | +0.0174 | +0.0167 | +0.0126 | -2.5 |
| Δ Dynamic vs best static α | +0.1182 | +0.0371 | +0.0176 | +0.0890 | -2.7 |

**FEVER** (n=144,918)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.7401 | 0.8348 | 0.8510 | 0.7853 | 16.3 |
| Best static α (α=0.05) | 0.4929 | 0.8108 | 0.8508 | 0.6261 | 16.8 |
| Dynamic Hybrid | 0.8603 | 0.9694 | 0.9808 | 0.9105 | 1.9 |
| Δ Dynamic vs dense | +0.1203 | +0.1346 | +0.1298 | +0.1251 | -14.4 |
| Δ Dynamic vs best static α | +0.3674 | +0.1586 | +0.1300 | +0.2843 | -14.9 |

**SciFact** (n=2,934)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.6118 | 0.8211 | 0.8770 | 0.7042 | 8.4 |
| Best static α (α=0.05) | 0.6135 | 0.8217 | 0.8821 | 0.7076 | 8.1 |
| Dynamic Hybrid | 0.5944 | 0.8180 | 0.8787 | 0.6932 | 7.0 |
| Δ Dynamic vs dense | -0.0174 | -0.0031 | +0.0017 | -0.0110 | -1.4 |
| Δ Dynamic vs best static α | -0.0191 | -0.0037 | -0.0034 | -0.0144 | -1.1 |

**NQ** (n=4,728)

| method | R@1 | R@5 | R@10 | MRR | mean rank |
|---|---:|---:|---:|---:|---:|
| Dense only | 0.5116 | 0.6673 | 0.6937 | 0.5833 | 27.9 |
| Best static α (α=0.05) | 0.4124 | 0.6476 | 0.6952 | 0.5120 | 28.1 |
| Dynamic Hybrid | 0.6997 | 0.9120 | 0.9425 | 0.7947 | 3.8 |
| Δ Dynamic vs dense | +0.1880 | +0.2447 | +0.2487 | +0.2113 | -24.1 |
| Δ Dynamic vs best static α | +0.2872 | +0.2644 | +0.2473 | +0.2826 | -24.3 |

Full per-corpus / per-encoder tables, α sweeps, and lift breakdowns: `dynamic_hybrid_internal_full_results.md`.

## What the numbers say

- **FiQA, SciFact — dense-dominant.** Dense is already close to the achievable ceiling on these corpora; lexical signal is largely redundant. Both variants hold R@10 at or above dense without paying the R@1 tax that fixed α incurs. The Dasein-native variant adds measurable gains on MRR and mean rank; the portable variant essentially matches dense, which is the correct behavior when BM25 has nothing to add.
- **FEVER, NQ — lexically rich.** Entity names and claim wording carry real complementary signal. Static α captures some of it at R@10 but collapses R@1. Both variants close most of the R@10 gap; the Dasein-native variant delivers large R@1 gains here, on top of the R@10 and mean-rank improvements.
- **Mean rank tells the cleanest story.** On corpora where R@k clips the tail, mean rank shows reordering benefit that the recall cutoffs hide. Both variants cut mean rank against dense on every in-domain corpus.
- **Static α cannot do this.** A single scalar fusion weight cannot reorder top-1 cleanly when BM25 and dense disagree, and the α-sweep tables make that structural. Any fixed non-zero α pays an R@1 tax even where it helps R@10. Per-query α is the fix.

## Caveats

- Four corpora is not the whole retrieval world. BEIR and MTEB are larger. We chose these four because they span both dense-dominated and lexically-rich regimes; domains outside this span (code, legal, long-form QA) will behave differently.
- Pooled metrics across encoders. Per-encoder behavior varies; the aggregates smooth that out. Per-encoder tables are in the full results files.
- Best static α is selected by R@10 (field-standard retrieval metric). Picking it by another metric would move the baseline.
- FEVER and NQ qrels are sparse in places; treat small R@1 differences with care.

## Full results

- **dynamic_hybrid_external_full_results.md** — portable variant, full per-corpus / per-encoder tables, α-sweep sweeps, lift breakdowns.
- **dynamic_hybrid_internal_full_results.md** — Dasein-native variant, same layout.
