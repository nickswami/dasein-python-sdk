# Dynamic Top-K — Dasein-native variant — full results

Dynamic Top-K predicts a per-query top-K cutoff on top of the Dasein-native Dynamic Hybrid pipeline. Same hybrid contract as before with one extra per-query scalar: the smallest top-K that retains the answer. All numbers below are at top-10 — the field-standard cutoff and the alternative every fixed-K stack would otherwise pick.

Three methods are compared throughout this document:

- **Dense (top-10)** — the no-system baseline. Pure dense retrieval at top-10.
- **Dynamic Hybrid (top-10)** — Dynamic Hybrid pipeline at fixed top-10.
- **Dynamic Hybrid + Dynamic Top-K** — Dynamic Hybrid pipeline with the per-query cutoff.

Δ rows are raw subtractions: negative on `avg records` / `avg tokens` means Dynamic Hybrid + Dynamic Top-K returns fewer; negative on `mean rank` means it pulls the answer closer to the top.

Cost columns:

- **avg records** — mean number of passages returned per query.
- **avg tokens** — mean total token count of the returned passages (`tiktoken cl100k_base`).

## 1. Aggregate headline — all eval queries pooled

Rollup over **223,763 eval queries**.

**Three methods at top-10** (n=223,763)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7606 | 0.8609 | 0.8771 | 0.8059 | 25.1 | 10.00 | 2859 |
| Dynamic Hybrid (top-10) | 0.8129 | 0.9468 | 0.9649 | 0.8727 | 8.0 | 10.00 | 2441 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8129 | 0.9396 | 0.9494 | 0.8697 | 10.9 | 3.65 | 905 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | +0.0523 | +0.0859 | +0.0878 | +0.0668 | -17.0 | +0.0% | -14.6% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | +0.0523 | +0.0787 | +0.0723 | +0.0638 | -14.1 | -63.5% | -68.4% |

## 2. Per-corpus breakdown

Each corpus pools every test query for that corpus across all encoders.

**FIQA** (n=7,525)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4639 | 0.6889 | 0.7733 | 0.5606 | 44.3 | 10.00 | 2017 |
| Dynamic Hybrid (top-10) | 0.4619 | 0.6803 | 0.7672 | 0.5548 | 45.4 | 10.00 | 1907 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4619 | 0.6756 | 0.7451 | 0.5516 | 49.4 | 7.70 | 1463 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0020 | -0.0086 | -0.0061 | -0.0057 | +1.1 | +0.0% | -5.4% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0020 | -0.0133 | -0.0282 | -0.0090 | +5.0 | -23.0% | -27.5% |

**FEVER** (n=205,660)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7805 | 0.8720 | 0.8847 | 0.8223 | 23.8 | 10.00 | 2936 |
| Dynamic Hybrid (top-10) | 0.8345 | 0.9610 | 0.9752 | 0.8914 | 6.1 | 10.00 | 2488 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8345 | 0.9539 | 0.9606 | 0.8885 | 8.8 | 3.39 | 877 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | +0.0540 | +0.0890 | +0.0906 | +0.0691 | -17.7 | +0.0% | -15.3% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | +0.0540 | +0.0819 | +0.0759 | +0.0662 | -15.0 | -66.1% | -70.1% |

**SCIFACT** (n=3,806)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6069 | 0.8198 | 0.8757 | 0.6968 | 24.1 | 10.00 | 3353 |
| Dynamic Hybrid (top-10) | 0.5972 | 0.8106 | 0.8734 | 0.6882 | 24.5 | 10.00 | 3278 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5972 | 0.7924 | 0.8261 | 0.6796 | 32.6 | 5.11 | 1655 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0097 | -0.0092 | -0.0024 | -0.0085 | +0.4 | +0.0% | -2.2% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0097 | -0.0273 | -0.0497 | -0.0171 | +8.5 | -48.9% | -50.6% |

**NQ** (n=6,772)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5713 | 0.7373 | 0.7634 | 0.6419 | 41.8 | 10.00 | 1164 |
| Dynamic Hybrid (top-10) | 0.6667 | 0.8861 | 0.9237 | 0.7620 | 16.2 | 10.00 | 1128 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6667 | 0.8808 | 0.9061 | 0.7589 | 19.4 | 6.17 | 700 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | +0.0954 | +0.1488 | +0.1602 | +0.1201 | -25.6 | +0.0% | -3.1% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | +0.0954 | +0.1435 | +0.1426 | +0.1170 | -22.4 | -38.3% | -39.9% |

## 3. Per-encoder breakdown

Per-encoder numbers across every split, at top-10.

### fiqa/test

#### fiqa/test — bge  (n=580)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4914 | 0.7276 | 0.8052 | 0.5905 | 38.4 | 10.00 | 2266 |
| Dynamic Hybrid (top-10) | 0.4810 | 0.6914 | 0.8069 | 0.5784 | 38.0 | 10.00 | 2070 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4810 | 0.6897 | 0.7845 | 0.5754 | 42.1 | 7.84 | 1640 |

#### fiqa/test — nomic  (n=560)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4107 | 0.6393 | 0.7446 | 0.5122 | 49.4 | 10.00 | 2284 |
| Dynamic Hybrid (top-10) | 0.3982 | 0.6411 | 0.7375 | 0.5018 | 50.6 | 10.00 | 2238 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3982 | 0.6375 | 0.7250 | 0.4998 | 52.8 | 8.44 | 1894 |

#### fiqa/test — gte  (n=642)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6044 | 0.8489 | 0.9159 | 0.7100 | 17.7 | 10.00 | 2062 |
| Dynamic Hybrid (top-10) | 0.5888 | 0.8349 | 0.8972 | 0.6930 | 21.3 | 10.00 | 1933 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5888 | 0.8255 | 0.8583 | 0.6866 | 28.3 | 6.08 | 1201 |

#### fiqa/test — e5  (n=583)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4511 | 0.6690 | 0.7427 | 0.5408 | 50.3 | 10.00 | 1651 |
| Dynamic Hybrid (top-10) | 0.3962 | 0.5883 | 0.6655 | 0.4817 | 64.4 | 10.00 | 1307 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3962 | 0.5883 | 0.6587 | 0.4809 | 65.6 | 8.99 | 1192 |

#### fiqa/test — e5mistral  (n=606)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5198 | 0.7393 | 0.8003 | 0.6110 | 39.1 | 10.00 | 1784 |
| Dynamic Hybrid (top-10) | 0.5396 | 0.7657 | 0.8267 | 0.6322 | 34.1 | 10.00 | 1789 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5396 | 0.7574 | 0.7954 | 0.6276 | 39.7 | 6.37 | 1164 |

#### fiqa/test — mxbai  (n=587)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4838 | 0.7172 | 0.8007 | 0.5869 | 39.3 | 10.00 | 2392 |
| Dynamic Hybrid (top-10) | 0.4838 | 0.6934 | 0.7990 | 0.5758 | 39.8 | 10.00 | 2141 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4838 | 0.6917 | 0.7836 | 0.5737 | 42.6 | 8.26 | 1794 |

#### fiqa/test — arctic  (n=578)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4637 | 0.6886 | 0.7855 | 0.5594 | 42.0 | 10.00 | 2502 |
| Dynamic Hybrid (top-10) | 0.4723 | 0.6938 | 0.7958 | 0.5654 | 40.0 | 10.00 | 2659 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4723 | 0.6869 | 0.7716 | 0.5615 | 44.4 | 7.18 | 1919 |

#### fiqa/test — qwen2  (n=621)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6200 | 0.8196 | 0.8953 | 0.7080 | 21.6 | 10.00 | 2190 |
| Dynamic Hybrid (top-10) | 0.6200 | 0.8406 | 0.8921 | 0.7105 | 22.0 | 10.00 | 2206 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6200 | 0.8229 | 0.8535 | 0.7038 | 29.2 | 5.57 | 1273 |

#### fiqa/test — minilm  (n=574)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.3868 | 0.6429 | 0.7317 | 0.4974 | 51.6 | 10.00 | 1766 |
| Dynamic Hybrid (top-10) | 0.4181 | 0.6429 | 0.7474 | 0.5150 | 48.6 | 10.00 | 1719 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4181 | 0.6429 | 0.7317 | 0.5131 | 51.5 | 8.79 | 1520 |

#### fiqa/test — jina3  (n=471)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.1847 | 0.3631 | 0.4650 | 0.2633 | 103.0 | 10.00 | 673 |
| Dynamic Hybrid (top-10) | 0.2442 | 0.4671 | 0.5669 | 0.3380 | 83.7 | 10.00 | 778 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2442 | 0.4650 | 0.5414 | 0.3346 | 88.2 | 8.60 | 682 |

### fiqa/test(held-out)

#### fiqa/test(held-out) — bgem3  (n=572)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4650 | 0.6836 | 0.7587 | 0.5570 | 46.8 | 10.00 | 2300 |
| Dynamic Hybrid (top-10) | 0.4545 | 0.6801 | 0.7657 | 0.5442 | 45.7 | 10.00 | 2222 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4545 | 0.6748 | 0.7343 | 0.5397 | 51.2 | 7.25 | 1625 |

#### fiqa/test(held-out) — bgebase  (n=577)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4263 | 0.6655 | 0.7643 | 0.5293 | 46.0 | 10.00 | 2225 |
| Dynamic Hybrid (top-10) | 0.4367 | 0.6551 | 0.7574 | 0.5315 | 47.2 | 10.00 | 2054 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4367 | 0.6551 | 0.7418 | 0.5297 | 50.0 | 8.24 | 1692 |

#### fiqa/test(held-out) — e5base  (n=574)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4390 | 0.6585 | 0.7578 | 0.5345 | 47.3 | 10.00 | 1878 |
| Dynamic Hybrid (top-10) | 0.3990 | 0.5749 | 0.6498 | 0.4725 | 67.3 | 10.00 | 1462 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.3990 | 0.5732 | 0.6446 | 0.4717 | 68.2 | 9.13 | 1349 |

### fever/test

#### fever/test — bge  (n=20,437)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8941 | 0.9699 | 0.9729 | 0.9300 | 6.5 | 10.00 | 3778 |
| Dynamic Hybrid (top-10) | 0.8941 | 0.9866 | 0.9907 | 0.9374 | 3.0 | 10.00 | 3300 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8941 | 0.9841 | 0.9857 | 0.9364 | 3.9 | 2.88 | 1016 |

#### fever/test — nomic  (n=505)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.1762 | 0.8317 | 0.1281 | 8.6 | 9.69 | 975 |
| Dynamic Hybrid (top-10) | 0.9881 | 1.0000 | 1.0000 | 0.9937 | 1.0 | 9.69 | 969 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.9881 | 1.0000 | 1.0000 | 0.9937 | 1.0 | 5.00 | 563 |

#### fever/test — gte  (n=20,600)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.9314 | 0.9751 | 0.9784 | 0.9517 | 5.4 | 10.00 | 3695 |
| Dynamic Hybrid (top-10) | 0.9407 | 0.9873 | 0.9912 | 0.9625 | 2.8 | 10.00 | 3009 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.9407 | 0.9840 | 0.9846 | 0.9612 | 4.1 | 2.30 | 794 |

#### fever/test — e5  (n=504)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0079 | 0.2282 | 0.8651 | 0.1457 | 8.0 | 9.51 | 1010 |
| Dynamic Hybrid (top-10) | 0.9663 | 0.9980 | 1.0000 | 0.9820 | 1.0 | 9.51 | 902 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.9663 | 0.9940 | 0.9940 | 0.9797 | 1.1 | 5.66 | 591 |

#### fever/test — e5mistral  (n=14,935)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6608 | 0.7324 | 0.7401 | 0.6934 | 52.3 | 10.00 | 2261 |
| Dynamic Hybrid (top-10) | 0.8782 | 0.9800 | 0.9888 | 0.9244 | 3.4 | 10.00 | 2011 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8782 | 0.9672 | 0.9693 | 0.9198 | 7.0 | 2.34 | 617 |

#### fever/test — mxbai  (n=20,428)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8983 | 0.9689 | 0.9720 | 0.9314 | 6.6 | 10.00 | 3247 |
| Dynamic Hybrid (top-10) | 0.8980 | 0.9867 | 0.9909 | 0.9395 | 2.9 | 10.00 | 2975 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8980 | 0.9838 | 0.9851 | 0.9383 | 4.0 | 3.00 | 1005 |

#### fever/test — arctic  (n=20,128)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8408 | 0.9167 | 0.9202 | 0.8766 | 17.0 | 10.00 | 4588 |
| Dynamic Hybrid (top-10) | 0.8820 | 0.9885 | 0.9922 | 0.9314 | 2.7 | 10.00 | 3823 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8820 | 0.9817 | 0.9829 | 0.9291 | 4.5 | 2.57 | 938 |

#### fever/test — qwen2  (n=16,394)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7017 | 0.7922 | 0.7990 | 0.7426 | 40.8 | 10.00 | 2231 |
| Dynamic Hybrid (top-10) | 0.8537 | 0.9765 | 0.9878 | 0.9087 | 3.6 | 10.00 | 2078 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8537 | 0.9542 | 0.9559 | 0.9012 | 9.6 | 2.23 | 626 |

#### fever/test — minilm  (n=19,447)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6162 | 0.8447 | 0.8837 | 0.7147 | 23.7 | 10.00 | 1820 |
| Dynamic Hybrid (top-10) | 0.6074 | 0.8651 | 0.9157 | 0.7183 | 17.7 | 10.00 | 1759 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6074 | 0.8561 | 0.8869 | 0.7134 | 23.0 | 5.76 | 1103 |

#### fever/test — jina3  (n=11,540)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.1002 | 0.1971 | 0.2358 | 0.1416 | 153.3 | 10.00 | 311 |
| Dynamic Hybrid (top-10) | 0.5404 | 0.8085 | 0.8721 | 0.6556 | 27.2 | 10.00 | 757 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5404 | 0.7974 | 0.8268 | 0.6485 | 35.9 | 5.37 | 536 |

### fever/test(held-out)

#### fever/test(held-out) — bgem3  (n=19,989)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8466 | 0.9461 | 0.9513 | 0.8920 | 10.7 | 10.00 | 2863 |
| Dynamic Hybrid (top-10) | 0.8370 | 0.9777 | 0.9865 | 0.9002 | 3.9 | 10.00 | 2340 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8370 | 0.9686 | 0.9717 | 0.8970 | 6.7 | 3.22 | 868 |

#### fever/test(held-out) — bgebase  (n=20,342)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8941 | 0.9678 | 0.9710 | 0.9286 | 6.8 | 10.00 | 3225 |
| Dynamic Hybrid (top-10) | 0.8907 | 0.9853 | 0.9908 | 0.9346 | 3.0 | 10.00 | 2918 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8907 | 0.9823 | 0.9847 | 0.9334 | 4.1 | 2.98 | 978 |

#### fever/test(held-out) — e5base  (n=20,411)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.8898 | 0.9679 | 0.9721 | 0.9262 | 6.5 | 10.00 | 2877 |
| Dynamic Hybrid (top-10) | 0.8272 | 0.9651 | 0.9786 | 0.8898 | 5.4 | 10.00 | 1488 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8272 | 0.9626 | 0.9715 | 0.8885 | 6.7 | 5.00 | 928 |

### scifact/test

#### scifact/test — bge  (n=293)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6348 | 0.8157 | 0.8976 | 0.7150 | 20.1 | 10.00 | 3504 |
| Dynamic Hybrid (top-10) | 0.6280 | 0.8191 | 0.9044 | 0.7119 | 18.9 | 10.00 | 3408 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6280 | 0.8055 | 0.8567 | 0.7041 | 27.2 | 5.40 | 1824 |

#### scifact/test — nomic  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5862 | 0.8069 | 0.8828 | 0.6836 | 23.2 | 10.00 | 3351 |
| Dynamic Hybrid (top-10) | 0.6138 | 0.8207 | 0.8759 | 0.6988 | 24.0 | 10.00 | 3356 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6138 | 0.8207 | 0.8586 | 0.6968 | 27.0 | 6.26 | 2079 |

#### scifact/test — gte  (n=300)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6933 | 0.9300 | 0.9600 | 0.7897 | 8.7 | 10.00 | 3362 |
| Dynamic Hybrid (top-10) | 0.6867 | 0.9033 | 0.9667 | 0.7844 | 7.7 | 10.00 | 3324 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6867 | 0.8800 | 0.9067 | 0.7733 | 18.0 | 3.82 | 1271 |

#### scifact/test — e5  (n=296)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6149 | 0.8108 | 0.8716 | 0.6995 | 24.6 | 10.00 | 3500 |
| Dynamic Hybrid (top-10) | 0.5777 | 0.7568 | 0.8074 | 0.6521 | 36.0 | 10.00 | 2994 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5777 | 0.7466 | 0.7804 | 0.6476 | 40.5 | 5.86 | 1784 |

#### scifact/test — e5mistral  (n=289)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6332 | 0.7993 | 0.8374 | 0.7027 | 31.0 | 10.00 | 3060 |
| Dynamic Hybrid (top-10) | 0.6609 | 0.8651 | 0.9204 | 0.7507 | 16.0 | 10.00 | 3073 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6609 | 0.8339 | 0.8651 | 0.7387 | 25.3 | 3.81 | 1146 |

#### scifact/test — mxbai  (n=293)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6416 | 0.8225 | 0.9010 | 0.7194 | 19.4 | 10.00 | 3432 |
| Dynamic Hybrid (top-10) | 0.6109 | 0.8020 | 0.8908 | 0.6987 | 21.1 | 10.00 | 3413 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6109 | 0.8020 | 0.8532 | 0.6935 | 27.6 | 5.70 | 1907 |

#### scifact/test — arctic  (n=297)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6330 | 0.8215 | 0.8754 | 0.7149 | 24.2 | 10.00 | 3618 |
| Dynamic Hybrid (top-10) | 0.5892 | 0.8114 | 0.8754 | 0.6869 | 24.1 | 10.00 | 3590 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5892 | 0.7980 | 0.8249 | 0.6780 | 32.9 | 4.12 | 1429 |

#### scifact/test — qwen2  (n=297)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6667 | 0.8923 | 0.9327 | 0.7562 | 13.8 | 10.00 | 3349 |
| Dynamic Hybrid (top-10) | 0.6566 | 0.8822 | 0.9394 | 0.7487 | 12.7 | 10.00 | 3329 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6566 | 0.8316 | 0.8418 | 0.7295 | 29.5 | 3.15 | 1024 |

#### scifact/test — minilm  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5241 | 0.7793 | 0.8241 | 0.6276 | 33.0 | 10.00 | 3236 |
| Dynamic Hybrid (top-10) | 0.5138 | 0.7690 | 0.8241 | 0.6159 | 33.1 | 10.00 | 3272 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5138 | 0.7517 | 0.7897 | 0.6098 | 39.0 | 6.17 | 1995 |

#### scifact/test — jina3  (n=289)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4844 | 0.7266 | 0.7820 | 0.5858 | 41.5 | 10.00 | 3188 |
| Dynamic Hybrid (top-10) | 0.4706 | 0.7093 | 0.7785 | 0.5802 | 42.3 | 10.00 | 3216 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4706 | 0.6955 | 0.7301 | 0.5720 | 50.5 | 5.30 | 1700 |

### scifact/test(held-out)

#### scifact/test(held-out) — bgem3  (n=288)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.5312 | 0.7604 | 0.8333 | 0.6325 | 32.1 | 10.00 | 3206 |
| Dynamic Hybrid (top-10) | 0.5486 | 0.7847 | 0.8368 | 0.6410 | 31.3 | 10.00 | 3199 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5486 | 0.7431 | 0.7604 | 0.6251 | 44.4 | 4.73 | 1521 |

#### scifact/test(held-out) — bgebase  (n=294)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6361 | 0.8537 | 0.8946 | 0.7218 | 20.5 | 10.00 | 3541 |
| Dynamic Hybrid (top-10) | 0.6156 | 0.8367 | 0.8946 | 0.7066 | 20.6 | 10.00 | 3488 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6156 | 0.8231 | 0.8605 | 0.7006 | 26.4 | 5.57 | 1917 |

#### scifact/test(held-out) — e5base  (n=290)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6034 | 0.8310 | 0.8862 | 0.7026 | 21.9 | 10.00 | 3220 |
| Dynamic Hybrid (top-10) | 0.5862 | 0.7724 | 0.8345 | 0.6662 | 31.2 | 10.00 | 2945 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5862 | 0.7655 | 0.8069 | 0.6613 | 35.9 | 6.56 | 1935 |

### nq/test

#### nq/test — bge  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0106 | 0.0106 | 0.0035 | 59.8 | 10.00 | 1178 |
| Dynamic Hybrid (top-10) | 0.8830 | 0.9894 | 1.0000 | 0.9344 | 1.2 | 10.00 | 886 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8830 | 0.9787 | 0.9787 | 0.9291 | 2.3 | 2.82 | 269 |

#### nq/test — nomic  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0106 | 0.0106 | 0.0027 | 61.6 | 10.00 | 1165 |
| Dynamic Hybrid (top-10) | 0.8830 | 1.0000 | 1.0000 | 0.9316 | 1.2 | 10.00 | 850 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8830 | 1.0000 | 1.0000 | 0.9316 | 1.2 | 3.59 | 332 |

#### nq/test — gte  (n=683)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7145 | 0.9165 | 0.9473 | 0.8024 | 11.4 | 10.00 | 1223 |
| Dynamic Hybrid (top-10) | 0.6940 | 0.9048 | 0.9370 | 0.7857 | 13.3 | 10.00 | 1159 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6940 | 0.8990 | 0.9107 | 0.7813 | 18.2 | 5.22 | 612 |

#### nq/test — e5  (n=94)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 60.9 | 10.00 | 1150 |
| Dynamic Hybrid (top-10) | 0.8617 | 1.0000 | 1.0000 | 0.9193 | 1.2 | 10.00 | 732 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8617 | 1.0000 | 1.0000 | 0.9193 | 1.2 | 4.52 | 409 |

#### nq/test — e5mistral  (n=562)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.3826 | 0.4502 | 0.4609 | 0.4113 | 106.6 | 10.00 | 890 |
| Dynamic Hybrid (top-10) | 0.7776 | 0.9644 | 0.9822 | 0.8631 | 4.7 | 10.00 | 900 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7776 | 0.9573 | 0.9662 | 0.8596 | 7.8 | 3.75 | 373 |

#### nq/test — mxbai  (n=684)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7091 | 0.9269 | 0.9459 | 0.8010 | 11.8 | 10.00 | 1335 |
| Dynamic Hybrid (top-10) | 0.6813 | 0.9211 | 0.9576 | 0.7858 | 9.6 | 10.00 | 1298 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6813 | 0.9196 | 0.9459 | 0.7841 | 11.8 | 6.73 | 880 |

#### nq/test — arctic  (n=670)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6925 | 0.8388 | 0.8537 | 0.7562 | 29.7 | 10.00 | 1192 |
| Dynamic Hybrid (top-10) | 0.7418 | 0.9418 | 0.9716 | 0.8316 | 6.9 | 10.00 | 1292 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7418 | 0.9358 | 0.9507 | 0.8281 | 10.7 | 5.89 | 777 |

#### nq/test — qwen2  (n=607)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.4843 | 0.5848 | 0.5914 | 0.5283 | 81.1 | 10.00 | 1342 |
| Dynamic Hybrid (top-10) | 0.7578 | 0.9621 | 0.9819 | 0.8477 | 4.9 | 10.00 | 1244 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7578 | 0.9341 | 0.9374 | 0.8376 | 13.3 | 3.39 | 436 |

#### nq/test — minilm  (n=675)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6148 | 0.8800 | 0.9200 | 0.7209 | 16.8 | 10.00 | 1152 |
| Dynamic Hybrid (top-10) | 0.5926 | 0.8815 | 0.9378 | 0.7158 | 13.5 | 10.00 | 1147 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5926 | 0.8800 | 0.9289 | 0.7145 | 15.1 | 8.17 | 934 |

#### nq/test — jina3  (n=565)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.1027 | 0.2283 | 0.3062 | 0.1588 | 135.4 | 10.00 | 282 |
| Dynamic Hybrid (top-10) | 0.2071 | 0.4460 | 0.5416 | 0.3077 | 90.1 | 10.00 | 395 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.2071 | 0.4442 | 0.5257 | 0.3056 | 93.0 | 8.71 | 353 |

### nq/test(held-out)

#### nq/test(held-out) — bgem3  (n=682)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7229 | 0.8827 | 0.9032 | 0.7891 | 20.3 | 10.00 | 1384 |
| Dynamic Hybrid (top-10) | 0.7287 | 0.9164 | 0.9472 | 0.8102 | 11.8 | 10.00 | 1326 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7287 | 0.9120 | 0.9282 | 0.8071 | 15.3 | 5.65 | 755 |

#### nq/test(held-out) — bgebase  (n=681)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.6799 | 0.8987 | 0.9280 | 0.7719 | 15.4 | 10.00 | 1331 |
| Dynamic Hybrid (top-10) | 0.6579 | 0.9148 | 0.9471 | 0.7650 | 11.7 | 10.00 | 1256 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6579 | 0.9148 | 0.9339 | 0.7633 | 14.1 | 6.95 | 881 |

#### nq/test(held-out) — e5base  (n=681)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7254 | 0.9163 | 0.9427 | 0.8102 | 12.5 | 10.00 | 1328 |
| Dynamic Hybrid (top-10) | 0.6931 | 0.9104 | 0.9530 | 0.7888 | 10.6 | 10.00 | 1238 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6931 | 0.9104 | 0.9471 | 0.7879 | 11.7 | 8.02 | 1007 |

## 4. Cost / quality lift per corpus — Dynamic Hybrid + Dynamic Top-K vs Dense (top-10)

Negative `record cut` / `token cut` = Dynamic Top-K returns fewer.

| Corpus | n | dyn-K R@10 | dense R@10 | ΔR@10 | dyn-K avg_n | record cut | token cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIQA | 7,525 | 0.7451 | 0.7733 | -0.0282 | 7.70 | -23.0% | -27.5% |
| FEVER | 205,660 | 0.9606 | 0.8847 | +0.0759 | 3.39 | -66.1% | -70.1% |
| SCIFACT | 3,806 | 0.8261 | 0.8757 | -0.0497 | 5.11 | -48.9% | -50.6% |
| NQ | 6,772 | 0.9061 | 0.7634 | +0.1426 | 6.17 | -38.3% | -39.9% |

## 4b. Cost / quality lift per corpus — Dynamic Hybrid + Dynamic Top-K vs Dynamic Hybrid (top-10)

The same-pipeline-without-Dynamic-Top-K comparison.

| Corpus | n | dyn-K R@10 | DH (top-10) R@10 | ΔR@10 | dyn-K avg_n | record cut | token cut |
|---|---:|---:|---:|---:|---:|---:|---:|
| FIQA | 7,525 | 0.7451 | 0.7672 | -0.0221 | 7.70 | -23.0% | -23.3% |
| FEVER | 205,660 | 0.9606 | 0.9752 | -0.0146 | 3.39 | -66.1% | -64.7% |
| SCIFACT | 3,806 | 0.8261 | 0.8734 | -0.0473 | 5.11 | -48.9% | -49.5% |
| NQ | 6,772 | 0.9061 | 0.9237 | -0.0176 | 6.17 | -38.3% | -38.0% |

## 5. Per-split detail — every method at every K

For completeness: Dense and Dynamic Hybrid at K=1, K=5, K=10, plus Dynamic Hybrid + Dynamic Top-K, on each split (in-training encoders and held-out encoders for each corpus).

### fiqa/test  (n=5,802)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.4700 | 0.4700 | 0.4700 | 0.4700 | 99.8 | 1.00 | 203 |
| Dense (top-5) | 0.4700 | 0.6948 | 0.6948 | 0.5556 | 58.4 | 5.00 | 1001 |
| Dense (top-10) | 0.4700 | 0.6948 | 0.7771 | 0.5666 | 43.6 | 10.00 | 1983 |
| Dynamic Hybrid (top-1) | 0.4714 | 0.4714 | 0.4714 | 0.4714 | 99.6 | 1.00 | 202 |
| Dynamic Hybrid (top-5) | 0.4714 | 0.6932 | 0.6932 | 0.5546 | 58.6 | 5.00 | 971 |
| Dynamic Hybrid (top-10) | 0.4714 | 0.6932 | 0.7799 | 0.5664 | 43.1 | 10.00 | 1906 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4714 | 0.6879 | 0.7565 | 0.5628 | 47.3 | 7.56 | 1436 |

### fiqa/test(held-out)  (n=1,723)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.4434 | 0.4434 | 0.4434 | 0.4434 | 104.7 | 1.00 | 208 |
| Dense (top-5) | 0.4434 | 0.6692 | 0.6692 | 0.5279 | 63.1 | 5.00 | 1054 |
| Dense (top-10) | 0.4434 | 0.6692 | 0.7603 | 0.5402 | 46.7 | 10.00 | 2134 |
| Dynamic Hybrid (top-1) | 0.4301 | 0.4301 | 0.4301 | 0.4301 | 107.2 | 1.00 | 194 |
| Dynamic Hybrid (top-5) | 0.4301 | 0.6367 | 0.6367 | 0.5041 | 69.2 | 5.00 | 947 |
| Dynamic Hybrid (top-10) | 0.4301 | 0.6367 | 0.7243 | 0.5161 | 53.4 | 10.00 | 1912 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.4301 | 0.6344 | 0.7069 | 0.5137 | 56.5 | 8.21 | 1556 |

### fever/test  (n=144,918)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.7401 | 0.7401 | 0.7401 | 0.7401 | 50.4 | 1.00 | 350 |
| Dense (top-5) | 0.7401 | 0.8348 | 0.8348 | 0.7809 | 32.6 | 5.00 | 1511 |
| Dense (top-10) | 0.7401 | 0.8348 | 0.8510 | 0.7831 | 30.4 | 10.00 | 2914 |
| Dynamic Hybrid (top-1) | 0.8273 | 0.8273 | 0.8273 | 0.8273 | 34.3 | 1.00 | 348 |
| Dynamic Hybrid (top-5) | 0.8273 | 0.9547 | 0.9547 | 0.8821 | 9.9 | 5.00 | 1360 |
| Dynamic Hybrid (top-10) | 0.8273 | 0.9547 | 0.9710 | 0.8844 | 6.9 | 10.00 | 2589 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8273 | 0.9466 | 0.9542 | 0.8810 | 10.1 | 3.24 | 857 |

### fever/test(held-out)  (n=60,742)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.8770 | 0.8770 | 0.8770 | 0.8770 | 24.7 | 1.00 | 388 |
| Dense (top-5) | 0.8770 | 0.9607 | 0.9607 | 0.9152 | 8.8 | 5.00 | 1566 |
| Dense (top-10) | 0.8770 | 0.9607 | 0.9649 | 0.9158 | 8.0 | 10.00 | 2989 |
| Dynamic Hybrid (top-1) | 0.8517 | 0.8517 | 0.8517 | 0.8517 | 29.6 | 1.00 | 367 |
| Dynamic Hybrid (top-5) | 0.8517 | 0.9760 | 0.9760 | 0.9070 | 5.8 | 5.00 | 1229 |
| Dynamic Hybrid (top-10) | 0.8517 | 0.9760 | 0.9853 | 0.9083 | 4.1 | 10.00 | 2247 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8517 | 0.9712 | 0.9760 | 0.9063 | 5.8 | 3.74 | 925 |

### scifact/test  (n=2,934)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.6118 | 0.6118 | 0.6118 | 0.6118 | 69.3 | 1.00 | 339 |
| Dense (top-5) | 0.6118 | 0.8211 | 0.8211 | 0.6922 | 33.1 | 5.00 | 1683 |
| Dense (top-10) | 0.6118 | 0.8211 | 0.8770 | 0.7000 | 23.9 | 10.00 | 3361 |
| Dynamic Hybrid (top-1) | 0.6012 | 0.6012 | 0.6012 | 0.6012 | 71.1 | 1.00 | 339 |
| Dynamic Hybrid (top-5) | 0.6012 | 0.8142 | 0.8142 | 0.6842 | 34.3 | 5.00 | 1649 |
| Dynamic Hybrid (top-10) | 0.6012 | 0.8142 | 0.8787 | 0.6932 | 23.5 | 10.00 | 3298 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6012 | 0.7969 | 0.8309 | 0.6847 | 31.7 | 4.95 | 1614 |

### scifact/test(held-out)  (n=872)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.5906 | 0.5906 | 0.5906 | 0.5906 | 72.8 | 1.00 | 336 |
| Dense (top-5) | 0.5906 | 0.8154 | 0.8154 | 0.6781 | 34.0 | 5.00 | 1658 |
| Dense (top-10) | 0.5906 | 0.8154 | 0.8716 | 0.6859 | 24.8 | 10.00 | 3323 |
| Dynamic Hybrid (top-1) | 0.5837 | 0.5837 | 0.5837 | 0.5837 | 73.9 | 1.00 | 333 |
| Dynamic Hybrid (top-5) | 0.5837 | 0.7982 | 0.7982 | 0.6636 | 37.2 | 5.00 | 1618 |
| Dynamic Hybrid (top-10) | 0.5837 | 0.7982 | 0.8555 | 0.6715 | 27.7 | 10.00 | 3212 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.5837 | 0.7775 | 0.8096 | 0.6626 | 35.5 | 5.62 | 1792 |

### nq/test  (n=4,728)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.5116 | 0.5116 | 0.5116 | 0.5116 | 86.3 | 1.00 | 112 |
| Dense (top-5) | 0.5116 | 0.6673 | 0.6673 | 0.5741 | 57.6 | 5.00 | 546 |
| Dense (top-10) | 0.5116 | 0.6673 | 0.6937 | 0.5777 | 52.9 | 10.00 | 1085 |
| Dynamic Hybrid (top-1) | 0.6552 | 0.6552 | 0.6552 | 0.6552 | 65.0 | 1.00 | 113 |
| Dynamic Hybrid (top-5) | 0.6552 | 0.8742 | 0.8742 | 0.7456 | 25.2 | 5.00 | 541 |
| Dynamic Hybrid (top-10) | 0.6552 | 0.8742 | 0.9126 | 0.7508 | 18.2 | 10.00 | 1065 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6552 | 0.8672 | 0.8930 | 0.7472 | 21.8 | 5.87 | 621 |

### nq/test(held-out)  (n=2,044)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-1) | 0.7094 | 0.7094 | 0.7094 | 0.7094 | 56.1 | 1.00 | 130 |
| Dense (top-5) | 0.7094 | 0.8992 | 0.8992 | 0.7870 | 20.7 | 5.00 | 666 |
| Dense (top-10) | 0.7094 | 0.8992 | 0.9247 | 0.7904 | 16.1 | 10.00 | 1348 |
| Dynamic Hybrid (top-1) | 0.6932 | 0.6932 | 0.6932 | 0.6932 | 59.0 | 1.00 | 126 |
| Dynamic Hybrid (top-5) | 0.6932 | 0.9139 | 0.9139 | 0.7834 | 17.8 | 5.00 | 641 |
| Dynamic Hybrid (top-10) | 0.6932 | 0.9139 | 0.9491 | 0.7880 | 11.3 | 10.00 | 1273 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.6932 | 0.9124 | 0.9364 | 0.7861 | 13.7 | 6.88 | 881 |

## 6. Bottom line

- **Dynamic Hybrid + Dynamic Top-K returns 63.5% fewer records and 68.4% fewer tokens than dense top-10**, with R@10 0.9494 vs 0.8771 (+0.0723). Pooled over 223,763 eval queries.
- Vs the same pipeline without Dynamic Top-K: 63.5% fewer records and 62.9% fewer tokens, R@10 -0.0155.
- **R@1 is preserved.** Dynamic Top-K always returns at least the rank-1 record; cuts come from trimming records 2–10 on queries where the answer is already in the retained prefix.
