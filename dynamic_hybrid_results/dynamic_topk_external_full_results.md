# Dynamic Top-K — portable variant — full results

Dynamic Top-K predicts a per-query top-K cutoff from the query alone. Two heads ride the same query encoder: one for dense retrieval (Dense + Dynamic Top-K) and one for the Dynamic Hybrid pipeline (Dynamic Hybrid + Dynamic Top-K). Both predict the smallest top-K that retains the answer. All numbers below are at top-10 — the field-standard cutoff and the alternative every fixed-K stack would otherwise pick.

Four methods are compared throughout this document:

- **Dense (top-10)** — the no-system baseline. Pure dense retrieval at top-10.
- **Dense + Dynamic Top-K** — the per-query cutoff applied to dense ranking.
- **Dynamic Hybrid (top-10)** — Dynamic Hybrid pipeline at fixed top-10.
- **Dynamic Hybrid + Dynamic Top-K** — Dynamic Hybrid pipeline with the per-query cutoff.

Δ rows are raw subtractions: negative on `avg records` / `avg tokens` means the Dynamic Top-K variant returns fewer; negative on `mean rank` means it pulls the answer closer to the top.

Cost columns:

- **avg records** — mean number of passages returned per query.
- **avg tokens** — mean total token count of the returned passages (`tiktoken cl100k_base`).

## 1. Aggregate headline — all eval queries pooled

Rollup over **239,395 eval queries**.

**Four methods at top-10** (n=239,395)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7109 | 0.8038 | 0.8162 | 0.7527 | 37.5 | 10.00 | 2756 |
| Dense + Dynamic Top-K | 0.7109 | 0.7991 | 0.8092 | 0.7510 | 38.8 | 6.91 | 1679 |
| Dynamic Hybrid (top-10) | 0.7107 | 0.8523 | 0.8788 | 0.7728 | 25.2 | 10.00 | 2617 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7107 | 0.8476 | 0.8717 | 0.7711 | 26.5 | 6.92 | 1545 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | -0.0048 | -0.0070 | -0.0016 | +1.3 | -30.9% | -39.1% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0002 | +0.0485 | +0.0625 | +0.0201 | -12.3 | +0.0% | -5.0% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0002 | +0.0438 | +0.0555 | +0.0185 | -11.0 | -30.8% | -43.9% |

## 2. Per-corpus breakdown

Each corpus pools every test query for that corpus across all encoders.

**FIQA** (n=7,521)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4642 | 0.6893 | 0.7737 | 0.5609 | 44.3 | 10.00 | 2018 |
| Dense + Dynamic Top-K | 0.4642 | 0.6893 | 0.7729 | 0.5608 | 44.4 | 9.96 | 2009 |
| Dynamic Hybrid (top-10) | 0.4578 | 0.6890 | 0.7741 | 0.5567 | 44.2 | 10.00 | 2005 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4578 | 0.6890 | 0.7733 | 0.5566 | 44.3 | 9.96 | 1997 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | +0.0000 | -0.0008 | -0.0001 | +0.1 | -0.4% | -0.4% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0064 | -0.0003 | +0.0004 | -0.0042 | -0.1 | +0.0% | -0.6% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0064 | -0.0003 | -0.0004 | -0.0043 | +0.1 | -0.4% | -1.0% |

**FEVER** (n=221,304)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7253 | 0.8094 | 0.8183 | 0.7635 | 37.2 | 10.00 | 2819 |
| Dense + Dynamic Top-K | 0.7253 | 0.8046 | 0.8113 | 0.7618 | 38.5 | 6.70 | 1665 |
| Dynamic Hybrid (top-10) | 0.7260 | 0.8612 | 0.8844 | 0.7855 | 24.2 | 10.00 | 2673 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7260 | 0.8563 | 0.8774 | 0.7838 | 25.5 | 6.72 | 1524 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | -0.0049 | -0.0070 | -0.0017 | +1.3 | -33.0% | -40.9% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | +0.0007 | +0.0517 | +0.0661 | +0.0220 | -13.0 | +0.0% | -5.2% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | +0.0007 | +0.0469 | +0.0591 | +0.0203 | -11.7 | -32.8% | -45.9% |

**SCIFACT** (n=3,806)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6069 | 0.8198 | 0.8757 | 0.6968 | 24.1 | 10.00 | 3353 |
| Dense + Dynamic Top-K | 0.6069 | 0.8098 | 0.8542 | 0.6926 | 27.7 | 8.78 | 2927 |
| Dynamic Hybrid (top-10) | 0.5996 | 0.8158 | 0.8757 | 0.6911 | 24.0 | 10.00 | 3248 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5996 | 0.8064 | 0.8547 | 0.6870 | 27.6 | 8.80 | 2829 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | -0.0100 | -0.0215 | -0.0042 | +3.6 | -12.2% | -12.7% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0074 | -0.0039 | +0.0000 | -0.0057 | -0.1 | +0.0% | -3.1% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0074 | -0.0134 | -0.0210 | -0.0098 | +3.5 | -12.0% | -15.6% |

**NQ** (n=6,764)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5720 | 0.7379 | 0.7642 | 0.6426 | 47.4 | 10.00 | 1163 |
| Dense + Dynamic Top-K | 0.5720 | 0.7346 | 0.7574 | 0.6414 | 48.6 | 9.25 | 1069 |
| Dynamic Hybrid (top-10) | 0.5532 | 0.7654 | 0.8125 | 0.6428 | 38.0 | 10.00 | 1086 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5532 | 0.7620 | 0.8053 | 0.6415 | 39.3 | 9.26 | 993 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | -0.0033 | -0.0068 | -0.0012 | +1.2 | -7.5% | -8.1% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0188 | +0.0275 | +0.0483 | +0.0002 | -9.4 | +0.0% | -6.6% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0188 | +0.0241 | +0.0411 | -0.0011 | -8.1 | -7.4% | -14.6% |

## 3. Per-encoder breakdown

Per-encoder numbers across every split, at top-10.

### fiqa/test

#### fiqa/test — bge  (n=580)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4914 | 0.7276 | 0.8052 | 0.5905 | 38.4 | 10.00 | 2266 |
| Dense + Dynamic Top-K | 0.4914 | 0.7276 | 0.8034 | 0.5903 | 38.7 | 9.95 | 2260 |
| Dynamic Hybrid (top-10) | 0.4776 | 0.7276 | 0.8034 | 0.5823 | 38.7 | 10.00 | 2264 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4776 | 0.7276 | 0.8017 | 0.5821 | 39.0 | 9.95 | 2257 |

#### fiqa/test — nomic  (n=559)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4114 | 0.6404 | 0.7460 | 0.5131 | 49.1 | 10.00 | 2285 |
| Dense + Dynamic Top-K | 0.4114 | 0.6404 | 0.7460 | 0.5131 | 49.1 | 10.00 | 2285 |
| Dynamic Hybrid (top-10) | 0.3846 | 0.6315 | 0.7388 | 0.4919 | 50.6 | 10.00 | 2206 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3846 | 0.6315 | 0.7388 | 0.4919 | 50.6 | 10.00 | 2206 |

#### fiqa/test — gte  (n=641)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6053 | 0.8502 | 0.9173 | 0.7111 | 17.4 | 10.00 | 2062 |
| Dense + Dynamic Top-K | 0.6053 | 0.8502 | 0.9126 | 0.7105 | 18.2 | 9.87 | 2035 |
| Dynamic Hybrid (top-10) | 0.6006 | 0.8518 | 0.9173 | 0.7081 | 17.4 | 10.00 | 2058 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6006 | 0.8518 | 0.9126 | 0.7075 | 18.2 | 9.87 | 2032 |

#### fiqa/test — e5  (n=584)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4503 | 0.6678 | 0.7414 | 0.5399 | 50.5 | 10.00 | 1651 |
| Dense + Dynamic Top-K | 0.4503 | 0.6678 | 0.7414 | 0.5399 | 50.5 | 10.00 | 1651 |
| Dynamic Hybrid (top-10) | 0.4435 | 0.6627 | 0.7397 | 0.5351 | 50.9 | 10.00 | 1638 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4435 | 0.6627 | 0.7397 | 0.5351 | 50.9 | 10.00 | 1638 |

#### fiqa/test — e5mistral  (n=606)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5198 | 0.7393 | 0.8003 | 0.6110 | 39.1 | 10.00 | 1784 |
| Dense + Dynamic Top-K | 0.5198 | 0.7393 | 0.8003 | 0.6110 | 39.1 | 10.00 | 1784 |
| Dynamic Hybrid (top-10) | 0.4950 | 0.7294 | 0.7937 | 0.5918 | 40.4 | 10.00 | 1748 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4950 | 0.7294 | 0.7937 | 0.5918 | 40.4 | 10.00 | 1748 |

#### fiqa/test — mxbai  (n=587)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4838 | 0.7172 | 0.8007 | 0.5869 | 39.3 | 10.00 | 2392 |
| Dense + Dynamic Top-K | 0.4838 | 0.7172 | 0.7990 | 0.5867 | 39.6 | 9.96 | 2386 |
| Dynamic Hybrid (top-10) | 0.4838 | 0.7172 | 0.8007 | 0.5860 | 39.3 | 10.00 | 2387 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4838 | 0.7172 | 0.7990 | 0.5858 | 39.7 | 9.96 | 2380 |

#### fiqa/test — arctic  (n=578)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4637 | 0.6886 | 0.7855 | 0.5594 | 42.0 | 10.00 | 2502 |
| Dense + Dynamic Top-K | 0.4637 | 0.6886 | 0.7855 | 0.5594 | 42.0 | 9.94 | 2489 |
| Dynamic Hybrid (top-10) | 0.4619 | 0.6869 | 0.7837 | 0.5577 | 42.3 | 10.00 | 2497 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4619 | 0.6869 | 0.7837 | 0.5577 | 42.3 | 9.95 | 2484 |

#### fiqa/test — qwen2  (n=621)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6200 | 0.8196 | 0.8953 | 0.7080 | 21.6 | 10.00 | 2190 |
| Dense + Dynamic Top-K | 0.6200 | 0.8196 | 0.8937 | 0.7077 | 21.9 | 9.92 | 2174 |
| Dynamic Hybrid (top-10) | 0.6055 | 0.8164 | 0.8921 | 0.6985 | 22.2 | 10.00 | 2176 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6055 | 0.8164 | 0.8905 | 0.6983 | 22.5 | 9.92 | 2160 |

#### fiqa/test — minilm  (n=574)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.3868 | 0.6429 | 0.7317 | 0.4974 | 51.6 | 10.00 | 1766 |
| Dense + Dynamic Top-K | 0.3868 | 0.6429 | 0.7317 | 0.4974 | 51.6 | 9.94 | 1755 |
| Dynamic Hybrid (top-10) | 0.3833 | 0.6411 | 0.7317 | 0.4952 | 51.6 | 10.00 | 1761 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3833 | 0.6411 | 0.7317 | 0.4952 | 51.6 | 9.94 | 1750 |

#### fiqa/test — jina3  (n=469)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.1855 | 0.3646 | 0.4670 | 0.2644 | 102.6 | 10.00 | 675 |
| Dense + Dynamic Top-K | 0.1855 | 0.3646 | 0.4670 | 0.2644 | 102.6 | 10.00 | 675 |
| Dynamic Hybrid (top-10) | 0.2047 | 0.3987 | 0.5011 | 0.2878 | 96.2 | 10.00 | 683 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2047 | 0.3987 | 0.5011 | 0.2878 | 96.2 | 10.00 | 683 |

### fiqa/test(held-out)

#### fiqa/test(held-out) — bgem3  (n=572)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4650 | 0.6836 | 0.7587 | 0.5570 | 46.8 | 10.00 | 2300 |
| Dense + Dynamic Top-K | 0.4650 | 0.6836 | 0.7587 | 0.5570 | 46.8 | 9.91 | 2278 |
| Dynamic Hybrid (top-10) | 0.4633 | 0.6818 | 0.7587 | 0.5553 | 46.8 | 10.00 | 2298 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4633 | 0.6818 | 0.7587 | 0.5553 | 46.8 | 9.91 | 2276 |

#### fiqa/test(held-out) — bgebase  (n=577)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4263 | 0.6655 | 0.7643 | 0.5293 | 46.0 | 10.00 | 2225 |
| Dense + Dynamic Top-K | 0.4263 | 0.6655 | 0.7643 | 0.5293 | 46.0 | 9.99 | 2222 |
| Dynamic Hybrid (top-10) | 0.4263 | 0.6655 | 0.7643 | 0.5290 | 46.0 | 10.00 | 2225 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4263 | 0.6655 | 0.7643 | 0.5290 | 46.0 | 9.99 | 2222 |

#### fiqa/test(held-out) — e5base  (n=573)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4398 | 0.6597 | 0.7592 | 0.5354 | 47.1 | 10.00 | 1880 |
| Dense + Dynamic Top-K | 0.4398 | 0.6597 | 0.7592 | 0.5354 | 47.1 | 10.00 | 1880 |
| Dynamic Hybrid (top-10) | 0.4415 | 0.6597 | 0.7592 | 0.5359 | 47.1 | 10.00 | 1877 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4415 | 0.6597 | 0.7592 | 0.5359 | 47.1 | 10.00 | 1877 |

### fever/test

#### fever/test — bge  (n=20,437)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8941 | 0.9699 | 0.9729 | 0.9300 | 6.5 | 10.00 | 3778 |
| Dense + Dynamic Top-K | 0.8941 | 0.9593 | 0.9596 | 0.9265 | 9.0 | 2.35 | 888 |
| Dynamic Hybrid (top-10) | 0.8903 | 0.9697 | 0.9729 | 0.9278 | 6.5 | 10.00 | 3776 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8903 | 0.9592 | 0.9595 | 0.9243 | 9.0 | 2.36 | 889 |

#### fever/test — nomic  (n=8,303)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1246 |
| Dense + Dynamic Top-K | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1246 |
| Dynamic Hybrid (top-10) | 0.3017 | 0.5135 | 0.5781 | 0.3897 | 86.2 | 10.00 | 641 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3017 | 0.5135 | 0.5781 | 0.3897 | 86.2 | 10.00 | 641 |

#### fever/test — gte  (n=20,601)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.9314 | 0.9750 | 0.9784 | 0.9517 | 5.4 | 10.00 | 3695 |
| Dense + Dynamic Top-K | 0.9314 | 0.9654 | 0.9654 | 0.9484 | 7.9 | 2.04 | 796 |
| Dynamic Hybrid (top-10) | 0.9302 | 0.9750 | 0.9784 | 0.9511 | 5.4 | 10.00 | 3695 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.9302 | 0.9653 | 0.9653 | 0.9477 | 7.9 | 2.04 | 797 |

#### fever/test — e5  (n=8,304)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1274 |
| Dense + Dynamic Top-K | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1274 |
| Dynamic Hybrid (top-10) | 0.2984 | 0.5031 | 0.5641 | 0.3845 | 89.0 | 10.00 | 718 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2984 | 0.5031 | 0.5641 | 0.3845 | 89.0 | 10.00 | 718 |

#### fever/test — e5mistral  (n=14,940)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6606 | 0.7322 | 0.7399 | 0.6932 | 52.4 | 10.00 | 2262 |
| Dense + Dynamic Top-K | 0.6606 | 0.7316 | 0.7371 | 0.6927 | 52.9 | 8.92 | 2013 |
| Dynamic Hybrid (top-10) | 0.4928 | 0.7086 | 0.7644 | 0.5847 | 47.9 | 10.00 | 1637 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4928 | 0.7081 | 0.7618 | 0.5843 | 48.3 | 8.95 | 1396 |

#### fever/test — mxbai  (n=20,429)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8982 | 0.9689 | 0.9719 | 0.9313 | 6.7 | 10.00 | 3247 |
| Dense + Dynamic Top-K | 0.8982 | 0.9579 | 0.9580 | 0.9277 | 9.3 | 2.40 | 843 |
| Dynamic Hybrid (top-10) | 0.8936 | 0.9685 | 0.9720 | 0.9286 | 6.6 | 10.00 | 3245 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8936 | 0.9575 | 0.9581 | 0.9250 | 9.3 | 2.42 | 844 |

#### fever/test — arctic  (n=20,130)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8407 | 0.9166 | 0.9201 | 0.8765 | 17.1 | 10.00 | 4588 |
| Dense + Dynamic Top-K | 0.8407 | 0.9074 | 0.9086 | 0.8735 | 19.3 | 3.93 | 1781 |
| Dynamic Hybrid (top-10) | 0.8187 | 0.9405 | 0.9550 | 0.8740 | 10.2 | 10.00 | 4128 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8187 | 0.9313 | 0.9435 | 0.8710 | 12.5 | 3.93 | 1322 |

#### fever/test — qwen2  (n=16,396)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7016 | 0.7921 | 0.7989 | 0.7425 | 40.9 | 10.00 | 2230 |
| Dense + Dynamic Top-K | 0.7016 | 0.7858 | 0.7893 | 0.7404 | 42.7 | 7.53 | 1631 |
| Dynamic Hybrid (top-10) | 0.5861 | 0.7837 | 0.8206 | 0.6707 | 36.9 | 10.00 | 1883 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5861 | 0.7778 | 0.8114 | 0.6688 | 38.6 | 7.57 | 1293 |

#### fever/test — minilm  (n=19,450)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6161 | 0.8445 | 0.8836 | 0.7146 | 23.8 | 10.00 | 1820 |
| Dense + Dynamic Top-K | 0.6161 | 0.8381 | 0.8705 | 0.7122 | 26.1 | 8.28 | 1519 |
| Dynamic Hybrid (top-10) | 0.6093 | 0.8439 | 0.8831 | 0.7104 | 23.8 | 10.00 | 1818 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6093 | 0.8374 | 0.8702 | 0.7080 | 26.2 | 8.32 | 1523 |

#### fever/test — jina3  (n=11,569)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0999 | 0.1966 | 0.2352 | 0.1412 | 153.4 | 10.00 | 310 |
| Dense + Dynamic Top-K | 0.0999 | 0.1966 | 0.2352 | 0.1412 | 153.4 | 10.00 | 310 |
| Dynamic Hybrid (top-10) | 0.2300 | 0.4717 | 0.5621 | 0.3314 | 88.9 | 10.00 | 545 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2300 | 0.4717 | 0.5621 | 0.3314 | 88.9 | 10.00 | 545 |

### fever/test(held-out)

#### fever/test(held-out) — bgem3  (n=19,990)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8465 | 0.9460 | 0.9512 | 0.8920 | 10.7 | 10.00 | 2863 |
| Dense + Dynamic Top-K | 0.8465 | 0.9451 | 0.9493 | 0.8916 | 11.1 | 8.12 | 2359 |
| Dynamic Hybrid (top-10) | 0.8020 | 0.9392 | 0.9490 | 0.8637 | 11.3 | 10.00 | 2824 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8020 | 0.9382 | 0.9461 | 0.8632 | 11.8 | 8.17 | 2336 |

#### fever/test(held-out) — bgebase  (n=20,344)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8940 | 0.9677 | 0.9709 | 0.9285 | 6.8 | 10.00 | 3224 |
| Dense + Dynamic Top-K | 0.8940 | 0.9677 | 0.9709 | 0.9285 | 6.8 | 9.92 | 3198 |
| Dynamic Hybrid (top-10) | 0.8889 | 0.9675 | 0.9709 | 0.9256 | 6.8 | 10.00 | 3222 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8889 | 0.9675 | 0.9709 | 0.9256 | 6.8 | 9.92 | 3197 |

#### fever/test(held-out) — e5base  (n=20,411)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8898 | 0.9679 | 0.9721 | 0.9262 | 6.5 | 10.00 | 2877 |
| Dense + Dynamic Top-K | 0.8898 | 0.9679 | 0.9721 | 0.9262 | 6.5 | 9.83 | 2831 |
| Dynamic Hybrid (top-10) | 0.8812 | 0.9679 | 0.9721 | 0.9216 | 6.6 | 10.00 | 2875 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8812 | 0.9679 | 0.9721 | 0.9216 | 6.6 | 9.85 | 2833 |

### scifact/test

#### scifact/test — bge  (n=293)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6348 | 0.8157 | 0.8976 | 0.7150 | 20.1 | 10.00 | 3504 |
| Dense + Dynamic Top-K | 0.6348 | 0.7986 | 0.8362 | 0.7048 | 30.5 | 6.74 | 2347 |
| Dynamic Hybrid (top-10) | 0.6348 | 0.8191 | 0.8976 | 0.7152 | 20.1 | 10.00 | 3503 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6348 | 0.8020 | 0.8362 | 0.7050 | 30.5 | 6.77 | 2356 |

#### scifact/test — nomic  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5862 | 0.8069 | 0.8828 | 0.6836 | 23.2 | 10.00 | 3351 |
| Dense + Dynamic Top-K | 0.5862 | 0.8069 | 0.8828 | 0.6836 | 23.2 | 10.00 | 3351 |
| Dynamic Hybrid (top-10) | 0.5828 | 0.7966 | 0.8621 | 0.6686 | 26.8 | 10.00 | 2922 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5828 | 0.7966 | 0.8621 | 0.6686 | 26.8 | 10.00 | 2922 |

#### scifact/test — gte  (n=300)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6933 | 0.9300 | 0.9600 | 0.7897 | 8.7 | 10.00 | 3362 |
| Dense + Dynamic Top-K | 0.6933 | 0.9000 | 0.9100 | 0.7773 | 17.4 | 5.46 | 1877 |
| Dynamic Hybrid (top-10) | 0.6933 | 0.9300 | 0.9600 | 0.7899 | 8.7 | 10.00 | 3361 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6933 | 0.8967 | 0.9067 | 0.7765 | 17.9 | 5.50 | 1889 |

#### scifact/test — e5  (n=296)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6149 | 0.8108 | 0.8716 | 0.6995 | 24.6 | 10.00 | 3500 |
| Dense + Dynamic Top-K | 0.6149 | 0.8108 | 0.8716 | 0.6995 | 24.6 | 10.00 | 3500 |
| Dynamic Hybrid (top-10) | 0.5642 | 0.7905 | 0.8615 | 0.6625 | 26.4 | 10.00 | 3014 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5642 | 0.7905 | 0.8615 | 0.6625 | 26.4 | 10.00 | 3014 |

#### scifact/test — e5mistral  (n=289)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6332 | 0.7993 | 0.8374 | 0.7027 | 31.0 | 10.00 | 3060 |
| Dense + Dynamic Top-K | 0.6332 | 0.7993 | 0.8374 | 0.7027 | 31.0 | 9.98 | 3055 |
| Dynamic Hybrid (top-10) | 0.5363 | 0.7370 | 0.8131 | 0.6254 | 35.1 | 10.00 | 2788 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5363 | 0.7370 | 0.8131 | 0.6254 | 35.1 | 9.98 | 2784 |

#### scifact/test — mxbai  (n=293)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6416 | 0.8225 | 0.9010 | 0.7194 | 19.4 | 10.00 | 3432 |
| Dense + Dynamic Top-K | 0.6416 | 0.7782 | 0.8225 | 0.7041 | 32.8 | 6.44 | 2188 |
| Dynamic Hybrid (top-10) | 0.6348 | 0.8225 | 0.9044 | 0.7164 | 18.8 | 10.00 | 3429 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6348 | 0.7816 | 0.8294 | 0.7018 | 31.6 | 6.51 | 2203 |

#### scifact/test — arctic  (n=297)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6330 | 0.8215 | 0.8754 | 0.7149 | 24.2 | 10.00 | 3618 |
| Dense + Dynamic Top-K | 0.6330 | 0.7912 | 0.8081 | 0.7022 | 35.2 | 6.76 | 2382 |
| Dynamic Hybrid (top-10) | 0.6364 | 0.8249 | 0.8956 | 0.7191 | 20.6 | 10.00 | 3530 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6364 | 0.8013 | 0.8350 | 0.7079 | 30.4 | 6.82 | 2316 |

#### scifact/test — qwen2  (n=297)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6667 | 0.8923 | 0.9327 | 0.7562 | 13.8 | 10.00 | 3349 |
| Dense + Dynamic Top-K | 0.6667 | 0.8889 | 0.9192 | 0.7541 | 16.2 | 9.62 | 3213 |
| Dynamic Hybrid (top-10) | 0.6700 | 0.8889 | 0.9293 | 0.7582 | 14.3 | 10.00 | 3319 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6700 | 0.8855 | 0.9192 | 0.7565 | 16.1 | 9.63 | 3188 |

#### scifact/test — minilm  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5241 | 0.7793 | 0.8241 | 0.6276 | 33.0 | 10.00 | 3236 |
| Dense + Dynamic Top-K | 0.5241 | 0.7759 | 0.8207 | 0.6267 | 33.5 | 9.64 | 3113 |
| Dynamic Hybrid (top-10) | 0.5241 | 0.7793 | 0.8241 | 0.6286 | 33.0 | 10.00 | 3227 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5241 | 0.7759 | 0.8207 | 0.6277 | 33.5 | 9.66 | 3111 |

#### scifact/test — jina3  (n=289)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4844 | 0.7266 | 0.7820 | 0.5858 | 41.5 | 10.00 | 3188 |
| Dense + Dynamic Top-K | 0.4844 | 0.7266 | 0.7820 | 0.5858 | 41.5 | 10.00 | 3187 |
| Dynamic Hybrid (top-10) | 0.5398 | 0.7647 | 0.8131 | 0.6358 | 35.6 | 10.00 | 3148 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5398 | 0.7647 | 0.8131 | 0.6358 | 35.6 | 9.99 | 3146 |

### scifact/test(held-out)

#### scifact/test(held-out) — bgem3  (n=288)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5312 | 0.7604 | 0.8333 | 0.6325 | 32.0 | 10.00 | 3206 |
| Dense + Dynamic Top-K | 0.5312 | 0.7604 | 0.8299 | 0.6321 | 32.6 | 9.72 | 3119 |
| Dynamic Hybrid (top-10) | 0.5312 | 0.7604 | 0.8368 | 0.6329 | 31.4 | 10.00 | 3204 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5312 | 0.7604 | 0.8299 | 0.6320 | 32.6 | 9.74 | 3125 |

#### scifact/test(held-out) — bgebase  (n=294)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6361 | 0.8537 | 0.8946 | 0.7218 | 20.5 | 10.00 | 3541 |
| Dense + Dynamic Top-K | 0.6361 | 0.8537 | 0.8946 | 0.7218 | 20.5 | 10.00 | 3540 |
| Dynamic Hybrid (top-10) | 0.6361 | 0.8537 | 0.8946 | 0.7218 | 20.5 | 10.00 | 3541 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6361 | 0.8537 | 0.8946 | 0.7218 | 20.5 | 10.00 | 3540 |

#### scifact/test(held-out) — e5base  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6034 | 0.8310 | 0.8862 | 0.7026 | 21.9 | 10.00 | 3220 |
| Dense + Dynamic Top-K | 0.6034 | 0.8310 | 0.8862 | 0.7026 | 21.9 | 10.00 | 3220 |
| Dynamic Hybrid (top-10) | 0.6034 | 0.8310 | 0.8862 | 0.7026 | 21.9 | 10.00 | 3220 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6034 | 0.8310 | 0.8862 | 0.7026 | 21.9 | 10.00 | 3220 |

### nq/test

#### nq/test — bge  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0106 | 0.0018 | 198.9 | 10.00 | 1127 |
| Dense + Dynamic Top-K | 0.0000 | 0.0000 | 0.0106 | 0.0018 | 198.9 | 9.49 | 1076 |
| Dynamic Hybrid (top-10) | 0.0000 | 0.0000 | 0.0106 | 0.0018 | 198.9 | 10.00 | 1127 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.0000 | 0.0000 | 0.0106 | 0.0018 | 198.9 | 9.52 | 1077 |

#### nq/test — nomic  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1133 |
| Dense + Dynamic Top-K | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1133 |
| Dynamic Hybrid (top-10) | 0.1702 | 0.3617 | 0.4894 | 0.2555 | 104.4 | 10.00 | 419 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.1702 | 0.3617 | 0.4894 | 0.2555 | 104.4 | 10.00 | 419 |

#### nq/test — gte  (n=683)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7145 | 0.9165 | 0.9473 | 0.8024 | 11.4 | 10.00 | 1223 |
| Dense + Dynamic Top-K | 0.7145 | 0.9034 | 0.9195 | 0.7975 | 16.4 | 7.22 | 890 |
| Dynamic Hybrid (top-10) | 0.6955 | 0.9122 | 0.9444 | 0.7886 | 12.0 | 10.00 | 1206 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6955 | 0.8990 | 0.9165 | 0.7837 | 17.0 | 7.25 | 878 |

#### nq/test — e5  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1138 |
| Dense + Dynamic Top-K | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 201.0 | 10.00 | 1138 |
| Dynamic Hybrid (top-10) | 0.1596 | 0.3511 | 0.4894 | 0.2491 | 104.5 | 10.00 | 424 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.1596 | 0.3511 | 0.4894 | 0.2491 | 104.5 | 10.00 | 424 |

#### nq/test — e5mistral  (n=562)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.3826 | 0.4502 | 0.4609 | 0.4113 | 106.6 | 10.00 | 890 |
| Dense + Dynamic Top-K | 0.3826 | 0.4502 | 0.4609 | 0.4113 | 106.6 | 9.96 | 886 |
| Dynamic Hybrid (top-10) | 0.2758 | 0.5036 | 0.5961 | 0.3678 | 80.1 | 10.00 | 625 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2758 | 0.5036 | 0.5961 | 0.3678 | 80.1 | 9.96 | 622 |

#### nq/test — mxbai  (n=684)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7091 | 0.9269 | 0.9459 | 0.8010 | 11.8 | 10.00 | 1335 |
| Dense + Dynamic Top-K | 0.7091 | 0.9240 | 0.9386 | 0.7996 | 13.2 | 9.31 | 1237 |
| Dynamic Hybrid (top-10) | 0.6754 | 0.9167 | 0.9430 | 0.7782 | 12.4 | 10.00 | 1314 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6754 | 0.9137 | 0.9357 | 0.7768 | 13.8 | 9.32 | 1216 |

#### nq/test — arctic  (n=668)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6946 | 0.8413 | 0.8563 | 0.7584 | 29.2 | 10.00 | 1193 |
| Dense + Dynamic Top-K | 0.6946 | 0.8323 | 0.8413 | 0.7554 | 31.8 | 7.89 | 931 |
| Dynamic Hybrid (top-10) | 0.6692 | 0.8368 | 0.8533 | 0.7415 | 29.8 | 10.00 | 1161 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6692 | 0.8278 | 0.8383 | 0.7384 | 32.4 | 7.94 | 904 |

#### nq/test — qwen2  (n=606)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4851 | 0.5858 | 0.5924 | 0.5291 | 80.9 | 10.00 | 1343 |
| Dense + Dynamic Top-K | 0.4851 | 0.5825 | 0.5842 | 0.5280 | 82.4 | 9.25 | 1231 |
| Dynamic Hybrid (top-10) | 0.3894 | 0.5776 | 0.6370 | 0.4679 | 72.3 | 10.00 | 999 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3894 | 0.5743 | 0.6287 | 0.4667 | 73.8 | 9.28 | 892 |

#### nq/test — minilm  (n=675)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6148 | 0.8800 | 0.9200 | 0.7209 | 16.8 | 10.00 | 1152 |
| Dense + Dynamic Top-K | 0.6148 | 0.8756 | 0.9111 | 0.7194 | 18.4 | 9.35 | 1078 |
| Dynamic Hybrid (top-10) | 0.6044 | 0.8696 | 0.9156 | 0.7130 | 17.7 | 10.00 | 1135 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6044 | 0.8637 | 0.9037 | 0.7107 | 19.8 | 9.37 | 1063 |

#### nq/test — jina3  (n=563)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.1030 | 0.2291 | 0.3073 | 0.1594 | 135.2 | 10.00 | 282 |
| Dense + Dynamic Top-K | 0.1030 | 0.2291 | 0.3073 | 0.1594 | 135.2 | 10.00 | 282 |
| Dynamic Hybrid (top-10) | 0.1776 | 0.4512 | 0.5702 | 0.2896 | 84.3 | 10.00 | 387 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.1776 | 0.4512 | 0.5702 | 0.2896 | 84.3 | 10.00 | 387 |

### nq/test(held-out)

#### nq/test(held-out) — bgem3  (n=680)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7250 | 0.8853 | 0.9059 | 0.7914 | 19.8 | 10.00 | 1384 |
| Dense + Dynamic Top-K | 0.7250 | 0.8853 | 0.9059 | 0.7914 | 19.8 | 9.55 | 1320 |
| Dynamic Hybrid (top-10) | 0.6956 | 0.8721 | 0.8941 | 0.7694 | 22.1 | 10.00 | 1342 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6956 | 0.8721 | 0.8926 | 0.7692 | 22.3 | 9.54 | 1280 |

#### nq/test(held-out) — bgebase  (n=680)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6809 | 0.9000 | 0.9294 | 0.7730 | 15.2 | 10.00 | 1330 |
| Dense + Dynamic Top-K | 0.6809 | 0.9000 | 0.9279 | 0.7728 | 15.4 | 10.00 | 1330 |
| Dynamic Hybrid (top-10) | 0.6779 | 0.8971 | 0.9309 | 0.7716 | 14.9 | 10.00 | 1330 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6779 | 0.8971 | 0.9294 | 0.7714 | 15.2 | 10.00 | 1330 |

#### nq/test(held-out) — e5base  (n=681)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7254 | 0.9163 | 0.9427 | 0.8102 | 12.4 | 10.00 | 1328 |
| Dense + Dynamic Top-K | 0.7254 | 0.9163 | 0.9427 | 0.8102 | 12.4 | 10.00 | 1328 |
| Dynamic Hybrid (top-10) | 0.7254 | 0.9163 | 0.9427 | 0.8100 | 12.4 | 10.00 | 1328 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7254 | 0.9163 | 0.9427 | 0.8100 | 12.4 | 10.00 | 1328 |

## 4. Cost / quality lift per corpus — Dynamic Hybrid + Dynamic Top-K vs Dense (top-10)

Negative `record cut` / `token cut` = the Dynamic Top-K variant returns fewer.

| Corpus | n | dyn-K R@10 | dense R@10 | ΔR@10 | dyn-K avg_n | record cut | token cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIQA | 7,521 | 0.7733 | 0.7737 | -0.0004 | 9.96 | -0.4% | -1.0% |
| FEVER | 221,304 | 0.8774 | 0.8183 | +0.0591 | 6.72 | -32.8% | -45.9% |
| SCIFACT | 3,806 | 0.8547 | 0.8757 | -0.0210 | 8.80 | -12.0% | -15.6% |
| NQ | 6,764 | 0.8053 | 0.7642 | +0.0411 | 9.26 | -7.4% | -14.6% |

## 4b. Cost / quality lift per corpus — Dense + Dynamic Top-K vs Dense (top-10)

Same dense ranking, just with a per-query cutoff instead of fixed top-10.

| Corpus | n | dyn-K R@10 | dense R@10 | ΔR@10 | dyn-K avg_n | record cut | token cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIQA | 7,521 | 0.7729 | 0.7737 | -0.0008 | 9.96 | -0.4% | -0.4% |
| FEVER | 221,304 | 0.8113 | 0.8183 | -0.0070 | 6.70 | -33.0% | -40.9% |
| SCIFACT | 3,806 | 0.8542 | 0.8757 | -0.0215 | 8.78 | -12.2% | -12.7% |
| NQ | 6,764 | 0.7574 | 0.7642 | -0.0068 | 9.25 | -7.5% | -8.1% |

## 4c. Cost / quality lift per corpus — Dynamic Hybrid + Dynamic Top-K vs Dynamic Hybrid (top-10)

The same-pipeline-without-Dynamic-Top-K comparison.

| Corpus | n | dyn-K R@10 | DH (top-10) R@10 | ΔR@10 | dyn-K avg_n | record cut | token cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIQA | 7,521 | 0.7733 | 0.7741 | -0.0008 | 9.96 | -0.4% | -0.4% |
| FEVER | 221,304 | 0.8774 | 0.8844 | -0.0070 | 6.72 | -32.8% | -43.0% |
| SCIFACT | 3,806 | 0.8547 | 0.8757 | -0.0210 | 8.80 | -12.0% | -12.9% |
| NQ | 6,764 | 0.8053 | 0.8125 | -0.0072 | 9.26 | -7.4% | -8.5% |

## 5. Per-split detail — every method at every K

For completeness: Dense and Dynamic Hybrid at K=1, K=5, K=10, plus Dense + Dynamic Top-K and Dynamic Hybrid + Dynamic Top-K, on each split (in-training encoders and held-out encoders for each corpus).

### fiqa/test  (n=5,799)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.4703 | 0.4703 | 0.4703 | 0.4703 | 99.8 | 1.00 | 203 |
| Dense (top-5) | 0.4703 | 0.6951 | 0.6951 | 0.5558 | 58.3 | 5.00 | 1001 |
| Dense (top-10) | 0.4703 | 0.6951 | 0.7775 | 0.5669 | 43.6 | 10.00 | 1983 |
| Dense + Dynamic Top-K | 0.4703 | 0.6951 | 0.7765 | 0.5668 | 43.7 | 9.96 | 1975 |
| Dynamic Hybrid (top-1) | 0.4620 | 0.4620 | 0.4620 | 0.4620 | 101.3 | 1.00 | 198 |
| Dynamic Hybrid (top-5) | 0.4620 | 0.6949 | 0.6949 | 0.5505 | 58.4 | 5.00 | 988 |
| Dynamic Hybrid (top-10) | 0.4620 | 0.6949 | 0.7781 | 0.5617 | 43.5 | 10.00 | 1967 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4620 | 0.6949 | 0.7770 | 0.5615 | 43.7 | 9.96 | 1959 |

### fiqa/test(held-out)  (n=1,722)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.4437 | 0.4437 | 0.4437 | 0.4437 | 104.6 | 1.00 | 208 |
| Dense (top-5) | 0.4437 | 0.6696 | 0.6696 | 0.5282 | 63.0 | 5.00 | 1054 |
| Dense (top-10) | 0.4437 | 0.6696 | 0.7607 | 0.5405 | 46.6 | 10.00 | 2135 |
| Dense + Dynamic Top-K | 0.4437 | 0.6696 | 0.7607 | 0.5405 | 46.6 | 9.97 | 2127 |
| Dynamic Hybrid (top-1) | 0.4437 | 0.4437 | 0.4437 | 0.4437 | 104.6 | 1.00 | 207 |
| Dynamic Hybrid (top-5) | 0.4437 | 0.6690 | 0.6690 | 0.5276 | 63.1 | 5.00 | 1053 |
| Dynamic Hybrid (top-10) | 0.4437 | 0.6690 | 0.7607 | 0.5400 | 46.6 | 10.00 | 2134 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4437 | 0.6690 | 0.7607 | 0.5400 | 46.6 | 9.97 | 2125 |

### fever/test  (n=160,559)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.6680 | 0.6680 | 0.6680 | 0.6680 | 66.3 | 1.00 | 328 |
| Dense (top-5) | 0.6680 | 0.7522 | 0.7522 | 0.7045 | 50.2 | 5.00 | 1425 |
| Dense (top-10) | 0.6680 | 0.7522 | 0.7628 | 0.7059 | 48.3 | 10.00 | 2755 |
| Dense + Dynamic Top-K | 0.6680 | 0.7456 | 0.7534 | 0.7037 | 50.0 | 5.72 | 1236 |
| Dynamic Hybrid (top-1) | 0.6761 | 0.6761 | 0.6761 | 0.6761 | 64.5 | 1.00 | 301 |
| Dynamic Hybrid (top-5) | 0.6761 | 0.8244 | 0.8244 | 0.7367 | 35.8 | 5.00 | 1315 |
| Dynamic Hybrid (top-10) | 0.6761 | 0.8244 | 0.8543 | 0.7407 | 30.2 | 10.00 | 2560 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6761 | 0.8179 | 0.8449 | 0.7385 | 32.0 | 5.73 | 1045 |

### fever/test(held-out)  (n=60,745)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.8770 | 0.8770 | 0.8770 | 0.8770 | 24.8 | 1.00 | 388 |
| Dense (top-5) | 0.8770 | 0.9606 | 0.9606 | 0.9151 | 8.8 | 5.00 | 1566 |
| Dense (top-10) | 0.8770 | 0.9606 | 0.9649 | 0.9157 | 8.0 | 10.00 | 2989 |
| Dense + Dynamic Top-K | 0.8770 | 0.9603 | 0.9642 | 0.9156 | 8.1 | 9.30 | 2798 |
| Dynamic Hybrid (top-1) | 0.8577 | 0.8577 | 0.8577 | 0.8577 | 28.5 | 1.00 | 379 |
| Dynamic Hybrid (top-5) | 0.8577 | 0.9583 | 0.9583 | 0.9031 | 9.3 | 5.00 | 1554 |
| Dynamic Hybrid (top-10) | 0.8577 | 0.9583 | 0.9641 | 0.9039 | 8.2 | 10.00 | 2974 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8577 | 0.9580 | 0.9631 | 0.9037 | 8.4 | 9.32 | 2791 |

### scifact/test  (n=2,934)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.6118 | 0.6118 | 0.6118 | 0.6118 | 69.3 | 1.00 | 339 |
| Dense (top-5) | 0.6118 | 0.8211 | 0.8211 | 0.6922 | 33.1 | 5.00 | 1683 |
| Dense (top-10) | 0.6118 | 0.8211 | 0.8770 | 0.7000 | 23.9 | 10.00 | 3361 |
| Dense + Dynamic Top-K | 0.6118 | 0.8081 | 0.8494 | 0.6946 | 28.5 | 8.45 | 2818 |
| Dynamic Hybrid (top-1) | 0.6022 | 0.6022 | 0.6022 | 0.6022 | 70.9 | 1.00 | 328 |
| Dynamic Hybrid (top-5) | 0.6022 | 0.8160 | 0.8160 | 0.6843 | 34.0 | 5.00 | 1618 |
| Dynamic Hybrid (top-10) | 0.6022 | 0.8160 | 0.8766 | 0.6926 | 23.9 | 10.00 | 3226 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6022 | 0.8037 | 0.8500 | 0.6873 | 28.3 | 8.47 | 2690 |

### scifact/test(held-out)  (n=872)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.5906 | 0.5906 | 0.5906 | 0.5906 | 72.7 | 1.00 | 336 |
| Dense (top-5) | 0.5906 | 0.8154 | 0.8154 | 0.6781 | 33.9 | 5.00 | 1658 |
| Dense (top-10) | 0.5906 | 0.8154 | 0.8716 | 0.6859 | 24.8 | 10.00 | 3323 |
| Dense + Dynamic Top-K | 0.5906 | 0.8154 | 0.8704 | 0.6858 | 25.0 | 9.91 | 3294 |
| Dynamic Hybrid (top-1) | 0.5906 | 0.5906 | 0.5906 | 0.5906 | 72.7 | 1.00 | 337 |
| Dynamic Hybrid (top-5) | 0.5906 | 0.8154 | 0.8154 | 0.6781 | 33.9 | 5.00 | 1659 |
| Dynamic Hybrid (top-10) | 0.5906 | 0.8154 | 0.8727 | 0.6861 | 24.6 | 10.00 | 3323 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5906 | 0.8154 | 0.8704 | 0.6858 | 25.0 | 9.91 | 3296 |

### nq/test  (n=4,723)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.5122 | 0.5122 | 0.5122 | 0.5122 | 94.5 | 1.00 | 112 |
| Dense (top-5) | 0.5122 | 0.6676 | 0.6676 | 0.5746 | 65.9 | 5.00 | 546 |
| Dense (top-10) | 0.5122 | 0.6676 | 0.6943 | 0.5782 | 61.0 | 10.00 | 1083 |
| Dense + Dynamic Top-K | 0.5122 | 0.6629 | 0.6847 | 0.5765 | 62.7 | 8.99 | 957 |
| Dynamic Hybrid (top-1) | 0.4899 | 0.4899 | 0.4899 | 0.4899 | 98.6 | 1.00 | 98 |
| Dynamic Hybrid (top-5) | 0.4899 | 0.7093 | 0.7093 | 0.5745 | 57.6 | 5.00 | 488 |
| Dynamic Hybrid (top-10) | 0.4899 | 0.7093 | 0.7650 | 0.5819 | 47.3 | 10.00 | 979 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4899 | 0.7044 | 0.7550 | 0.5801 | 49.0 | 9.02 | 855 |

### nq/test(held-out)  (n=2,041)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.7104 | 0.7104 | 0.7104 | 0.7104 | 55.9 | 1.00 | 130 |
| Dense (top-5) | 0.7104 | 0.9005 | 0.9005 | 0.7881 | 20.4 | 5.00 | 666 |
| Dense (top-10) | 0.7104 | 0.9005 | 0.9260 | 0.7916 | 15.8 | 10.00 | 1348 |
| Dense + Dynamic Top-K | 0.7104 | 0.9005 | 0.9255 | 0.7915 | 15.9 | 9.85 | 1326 |
| Dynamic Hybrid (top-1) | 0.6997 | 0.6997 | 0.6997 | 0.6997 | 58.0 | 1.00 | 128 |
| Dynamic Hybrid (top-5) | 0.6997 | 0.8951 | 0.8951 | 0.7799 | 21.4 | 5.00 | 656 |
| Dynamic Hybrid (top-10) | 0.6997 | 0.8951 | 0.9226 | 0.7837 | 16.5 | 10.00 | 1333 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6997 | 0.8951 | 0.9216 | 0.7836 | 16.6 | 9.84 | 1313 |

## 6. Bottom line

- **Dynamic Hybrid + Dynamic Top-K returns 30.8% fewer records and 43.9% fewer tokens than dense top-10**, with R@10 0.8717 vs 0.8162 (+0.0555). Pooled over 239,395 eval queries.
- Vs the same pipeline without Dynamic Top-K: 30.8% fewer records and 41.0% fewer tokens, R@10 -0.0071.
- Dense + Dynamic Top-K alone (no Dynamic Hybrid pipeline): 30.9% fewer records and 39.1% fewer tokens than dense top-10, R@10 -0.0070.
- **R@1 is preserved.** Dynamic Top-K always returns at least the rank-1 record; cuts come from trimming records 2–10 on queries where the answer is already in the retained prefix.
