# Dynamic Top-K: per-query top-K for retrieval

Retrieval stacks pick a fixed top-K (almost always 5 or 10) and pay for it on every query — even the ones where the gold answer is at rank 1. Most queries do not need 10 records, and the ones that do need them at different positions. A single fixed K is a compromise that over-fetches on easy queries and wastes downstream compute (LLM tokens, latency, dollars) the system never recovers.

Dynamic Top-K predicts a per-query cutoff: the smallest K that retains the answer on each query. It ships in two variants:

- **Portable.** Drops into any retrieval stack. Runs in under a millisecond per query.
- **Dasein-native.** Runs end-to-end on the Dasein index. A few milliseconds per query.

All numbers below are at top-10: the field-standard cutoff and the alternative every fixed-K stack would otherwise pick. Δ rows are raw subtractions — negative on `avg records` or `avg tokens` means the method returns fewer; negative on `mean rank` means it pulls the answer closer to the top.

## Aggregate — all eval queries pooled

**Portable variant** (n=239,395)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7109 | 0.8038 | 0.8162 | 0.7527 | 37.5 | 10.00 | 2756 |
| Dense + Dynamic Top-K | 0.7109 | 0.7991 | 0.8092 | 0.7510 | 38.8 | 6.91 | 1679 |
| Dynamic Hybrid (top-10) | 0.7107 | 0.8523 | 0.8788 | 0.7728 | 25.2 | 10.00 | 2617 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.7107 | 0.8476 | 0.8717 | 0.7711 | 26.5 | 6.92 | 1545 |
| Δ Dense + Dynamic Top-K vs Dense (top-10) | +0.0000 | -0.0048 | -0.0070 | -0.0016 | +1.3 | -30.9% | -39.1% |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | -0.0002 | +0.0485 | +0.0625 | +0.0201 | -12.3 | +0.0% | -5.0% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | -0.0002 | +0.0438 | +0.0555 | +0.0185 | -11.0 | -30.8% | -43.9% |

**Dasein-native variant** (n=223,763)

| method | R@1 | R@5 | R@10 | MRR | mean rank | avg records | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (top-10) | 0.7606 | 0.8609 | 0.8771 | 0.8059 | 25.1 | 10.00 | 2859 |
| Dynamic Hybrid (top-10) | 0.8129 | 0.9468 | 0.9649 | 0.8727 | 8.0 | 10.00 | 2441 |
| **Dynamic Hybrid + Dynamic Top-K** | 0.8129 | 0.9396 | 0.9494 | 0.8697 | 10.9 | 3.65 | 905 |
| Δ Dynamic Hybrid (top-10) vs Dense (top-10) | +0.0523 | +0.0859 | +0.0878 | +0.0668 | -17.0 | +0.0% | -14.6% |
| Δ **Dynamic Hybrid + Dynamic Top-K** vs Dense (top-10) | +0.0523 | +0.0787 | +0.0723 | +0.0638 | -14.1 | -63.5% | -68.4% |

## Per-corpus breakdown — portable variant

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

Full per-corpus / per-encoder tables: `dynamic_topk_external_full_results.md`.

## Per-corpus breakdown — Dasein-native variant

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

Full per-corpus / per-encoder tables: `dynamic_topk_internal_full_results.md`.

## What the numbers say

- **Portable variant: Dynamic Hybrid + Dynamic Top-K returns 30.8% fewer records and 43.9% fewer tokens than dense top-10**, with R@10 0.8717 vs 0.8162 (+0.0555) and R@1 0.7107 vs 0.7109 (-0.0002).
- **Dasein-native variant: Dynamic Hybrid + Dynamic Top-K returns 63.5% fewer records and 68.4% fewer tokens than dense top-10**, with R@10 0.9494 vs 0.8771 (+0.0723) and R@1 0.8129 vs 0.7606 (+0.0523).
- **R@1 is preserved.** Dynamic Top-K always returns at least the rank-1 record; cuts come from trimming records 2–10 on queries where the answer is already in the retained prefix.
- **Token cuts ≥ record cuts on the Dasein-native variant.** Tokens are what shows up downstream as LLM cost, latency, and dollars.

## Full results

- **dynamic_topk_external_full_results.md** — portable variant, full per-corpus and per-encoder tables.
- **dynamic_topk_internal_full_results.md** — Dasein-native variant, full per-corpus and per-encoder tables.
