# Dataset Preprocessing and Statistics

The following files are used to process the input data so that statistics may be performed prior to the implementation of black box machine learning models. 
The purpose of these files is to provide the developer domain insights as to how hate speech differs from non-hate speech and how hate speech compares across multiple languages.

| Dataset |  Num NS  | Num HS | % HS |
|:-----|:--------:|:------:|:------:|
| arabic_mulki   | 3649 | 468 | 11% |
| danish | 2850 | 425 | 13% |
| english_basile | 7530 | 5470 | 42% |
| english_davidson | 4163 | 1430 | 26% |
| english_founta | 34487 | 2075 | 6% |
| english_gilbert | 9507 | 1196 | 11% |
| english_ousidhoum | 4369 | 1278 | 23% |
| english_waseem | 7679 | 2759 | 26%  |
| french_ousidhoum | 821 | 207 | 20% |
| german_bertschneider | 5141 | 1331 | 21% |
| german_ross | 364 | 105 | 22% |
| indonesian_alfina | 453 | 260 | 36% |
| indonesian_ibrohim | 7607 | 5561 | 42% |
| italian_bosco | 4071 | 2766 | 40% |
| italian_manuel | 4436 | 843 | 16% |
| portuguese | 4440 | 1228 | 22% |
| spanish_basile | 3861 | 2739 | 42% |
| spanish_pereira | 4433 | 1567 | 26% |
| turkish | 28035 | 6757 | 19% |

## Word Count Statistics (wordcount.ipynb & pos.ipynb)

This portion of the project begins with a simple word count analysis comparing hate speech versus non hate speech for each language. If the language specific dataset offers additional taxonomies with labelings then statistics will be performed on these labelings as well to improve understanding of the taxonomy definitions. We continue this work by examining if there are statistically significant differences in the usage of parts of speech between the hate speech and normal examples.

Lift is the percentage change from POS % in NS to POS % in HS.

We do not have language specific POS embeddings for the following languages: [Arabic, Indonesian, Turkish], and thus english pos embeddings are used in place.


### Arabic (Mulki)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 1756 | 268 | 3.18% | 3.44% | 8.36% | 0.111308 |
| adposition | 1083 | 138 | 1.96% | 1.77% | -9.24% | 0.130791 |
| adverb | 1056 | 150 | 1.91% | 1.93% | 1.11% | 0.457431 |
| auxiliary | 139 | 21 | 0.25% | 0.27% | 11.22% | 0.343910 |
| coordinating conjunction | 280 | 25 | 0.51% | 0.32% | -34.51% | 0.012281 |
| determiner | 465 | 46 | 0.84% | 0.59% | -28.61% | 0.009282 |
| interjection | 329 | 41 | 0.60% | 0.53% | -9.92% | 0.245286 |
| noun | 6352 | 925 | 11.52% | 11.87% | 3.16% | 0.175823 |
| numeral | 115 | 12 | 0.21% | 0.15% | -20.68% | 0.185994 |
| particle | 224 | 21 | 0.41% | 0.27% | -30.80% | 0.036391 |
| pronoun | 595 | 84 | 1.08% | 1.08% | 0.94% | 0.479034 |
| proper noun | 31865 | 4632 | 57.78% | 59.46% | 2.90% | 0.002447 |
| punctuation | 2986 | 301 | 5.41% | 3.86% | -28.44% | 0.000000 |
| symbol | 2506 | 412 | 4.54% | 5.29% | 16.60% | 0.001878 |
| verb | 3323 | 510 | 6.03% | 6.55% | 8.81% | 0.035549 |

### Danish
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 4107 | 854 | 11.03% | 10.83% | -1.70% | 0.311385 |
| adposition | 608 | 117 | 1.63% | 1.48% | -8.49% | 0.178858 |
| adverb | 3985 | 825 | 10.70% | 10.47% | -2.13% | 0.272866 |
| auxiliary | 482 | 130 | 1.29% | 1.65% | 28.09% | 0.007111 |
| coordinating conjunction | 110 | 26 | 0.30% | 0.33% | 14.88% | 0.276416 |
| determiner | 509 | 122 | 1.37% | 1.55% | 13.90% | 0.102496 |
| interjection | 154 | 38 | 0.41% | 0.48% | 18.83% | 0.180967 |
| noun | 10220 | 2244 | 27.45% | 28.47% | 3.74% | 0.032692 |
| numeral | 649 | 100 | 1.74% | 1.27% | -26.61% | 0.001140 |
| particle | 9 | 5 | 0.02% | 0.06% | 183.37% | 0.033877 |
| pronoun | 395 | 87 | 1.06% | 1.10% | 4.95% | 0.349967 |
| proper noun | 1635 | 290 | 4.39% | 3.68% | -15.99% | 0.002067 |
| punctuation | 6859 | 1445 | 18.42% | 18.33% | -0.45% | 0.430013 |
| subordinating conjunction | 244 | 59 | 0.66% | 0.75% | 15.66% | 0.166305 |
| symbol | 314 | 30 | 0.84% | 0.38% | -53.52% | 0.000002 |
| verb | 5173 | 1153 | 13.89% | 14.63% | 5.34% | 0.043682 |

### English (Basile)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 11987 | 9393 | 7.90% | 8.52% | 7.85% | 0.000000 |
| adposition | 4583 | 3341 | 3.02% | 3.03% | 0.34% | 0.439580 |
| adverb | 4131 | 3367 | 2.72% | 3.05% | 12.19% | 0.000000 |
| auxiliary | 2289 | 2109 | 1.51% | 1.91% | 26.82% | 0.000000 |
| coordinating conjunction | 744 | 568 | 0.49% | 0.52% | 5.12% | 0.185291 |
| determiner | 605 | 606 | 0.40% | 0.55% | 37.86% | 0.000000 |
| interjection | 840 | 611 | 0.55% | 0.55% | 0.16% | 0.489254 |
| noun | 43893 | 33276 | 28.93% | 30.19% | 4.34% | 0.000000 |
| numeral | 2339 | 1430 | 1.54% | 1.30% | -15.83% | 0.000000 |
| particle | 2078 | 1688 | 1.37% | 1.53% | 11.82% | 0.000307 |
| pronoun | 1342 | 1037 | 0.88% | 0.94% | 6.38% | 0.066962 |
| proper noun | 18310 | 11283 | 12.07% | 10.24% | -15.18% | 0.000000 |
| punctuation | 29613 | 18737 | 19.52% | 17.00% | -12.91% | 0.000000 |
| subordinating conjunction | 184 | 134 | 0.12% | 0.12% | 0.44% | 0.487015 |
| symbol | 6719 | 5690 | 4.43% | 5.16% | 16.56% | 0.000000 |
| verb | 20082 | 15895 | 13.24% | 14.42% | 8.94% | 0.000000 |

### English (Davidson)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 4400 | 1561 | 6.33% | 7.65% | 20.82% | 0.000000 |
| adposition | 2895 | 921 | 4.17% | 4.51% | 8.38% | 0.015657 |
| adverb | 1555 | 410 | 2.24% | 2.01% | -10.09% | 0.024548 |
| auxiliary | 1099 | 377 | 1.58% | 1.85% | 16.98% | 0.004551 |
| coordinating conjunction | 2293 | 572 | 3.30% | 2.80% | -14.97% | 0.000163 |
| determiner | 293 | 157 | 0.42% | 0.77% | 82.94% | 0.000000 |
| interjection | 413 | 155 | 0.59% | 0.76% | 28.27% | 0.004769 |
| noun | 15605 | 4851 | 22.46% | 23.77% | 5.83% | 0.000046 |
| numeral | 3030 | 675 | 4.36% | 3.31% | -24.08% | 0.000000 |
| particle | 967 | 341 | 1.39% | 1.67% | 20.27% | 0.001857 |
| pronoun | 437 | 225 | 0.63% | 1.10% | 75.64% | 0.000000 |
| proper noun | 10363 | 2860 | 14.92% | 14.02% | -6.03% | 0.000670 |
| punctuation | 14875 | 4098 | 21.41% | 20.08% | -6.20% | 0.000020 |
| subordinating conjunction | 81 | 19 | 0.12% | 0.09% | -16.97% | 0.209414 |
| symbol | 3468 | 682 | 4.99% | 3.34% | -32.98% | 0.000000 |
| verb | 7097 | 2369 | 10.22% | 11.61% | 13.66% | 0.000000 |

### English (Founta)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 45024 | 2689 | 6.86% | 7.49% | 9.18% | 0.000003 |
| adposition | 20329 | 1337 | 3.10% | 3.72% | 20.27% | 0.000000 |
| adverb | 18316 | 955 | 2.79% | 2.66% | -4.62% | 0.071364 |
| auxiliary | 10982 | 821 | 1.67% | 2.29% | 36.77% | 0.000000 |
| coordinating conjunction | 3443 | 181 | 0.52% | 0.50% | -3.43% | 0.314022 |
| determiner | 2330 | 232 | 0.36% | 0.65% | 82.67% | 0.000000 |
| interjection | 4260 | 245 | 0.65% | 0.68% | 5.51% | 0.213547 |
| noun | 173718 | 8512 | 26.47% | 23.71% | -10.45% | 0.000000 |
| numeral | 15171 | 448 | 2.31% | 1.25% | -45.92% | 0.000000 |
| particle | 8812 | 736 | 1.34% | 2.05% | 52.83% | 0.000000 |
| pronoun | 5039 | 429 | 0.77% | 1.19% | 55.92% | 0.000000 |
| proper noun | 86363 | 5397 | 13.16% | 15.03% | 14.22% | 0.000000 |
| punctuation | 159225 | 8185 | 24.26% | 22.80% | -6.05% | 0.000000 |
| subordinating conjunction | 889 | 55 | 0.14% | 0.15% | 14.99% | 0.170340 |
| symbol | 21174 | 664 | 3.23% | 1.85% | -42.61% | 0.000000 |
| verb | 74632 | 4705 | 11.37% | 13.10% | 15.23% | 0.000000 |

### English (Gilbert)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 11269 | 2184 | 10.77% | 12.77% | 18.70% | 0.000000 |
| adposition | 1201 | 211 | 1.15% | 1.23% | 7.98% | 0.156206 |
| adverb | 4624 | 834 | 4.42% | 4.88% | 10.53% | 0.003610 |
| auxiliary | 2570 | 451 | 2.46% | 2.64% | 7.63% | 0.075404 |
| coordinating conjunction | 299 | 31 | 0.29% | 0.18% | -34.70% | 0.006635 |
| determiner | 369 | 79 | 0.35% | 0.46% | 32.37% | 0.014533 |
| interjection | 988 | 115 | 0.94% | 0.67% | -28.19% | 0.000183 |
| noun | 29756 | 5065 | 28.43% | 29.63% | 4.23% | 0.000655 |
| numeral | 2858 | 239 | 2.73% | 1.40% | -48.61% | 0.000000 |
| particle | 1519 | 270 | 1.45% | 1.58% | 9.15% | 0.095186 |
| pronoun | 1232 | 192 | 1.18% | 1.12% | -4.17% | 0.282782 |
| proper noun | 8920 | 1265 | 8.52% | 7.40% | -13.12% | 0.000000 |
| punctuation | 20932 | 2884 | 20.00% | 16.87% | -15.62% | 0.000000 |
| subordinating conjunction | 250 | 43 | 0.24% | 0.25% | 7.32% | 0.348010 |
| symbol | 908 | 81 | 0.87% | 0.47% | -44.77% | 0.000000 |
| verb | 16239 | 3100 | 15.51% | 18.13% | 16.90% | 0.000000 |

### English (Ousidhoum)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 4236 | 1249 | 8.47% | 8.71% | 2.81% | 0.186279 |
| adposition | 4801 | 1278 | 9.60% | 8.91% | -7.19% | 0.006007 |
| adverb | 1610 | 504 | 3.22% | 3.51% | 9.23% | 0.040984 |
| auxiliary | 1037 | 305 | 2.07% | 2.13% | 2.73% | 0.342502 |
| coordinating conjunction | 191 | 31 | 0.38% | 0.22% | -41.92% | 0.001047 |
| determiner | 218 | 46 | 0.44% | 0.32% | -25.21% | 0.028749 |
| interjection | 433 | 128 | 0.87% | 0.89% | 3.58% | 0.369607 |
| noun | 16767 | 4757 | 33.54% | 33.17% | -1.12% | 0.199807 |
| numeral | 805 | 226 | 1.61% | 1.58% | -1.86% | 0.395251 |
| particle | 719 | 194 | 1.44% | 1.35% | -5.62% | 0.229402 |
| pronoun | 494 | 136 | 0.99% | 0.95% | -3.55% | 0.346106 |
| proper noun | 6147 | 1803 | 12.30% | 12.57% | 2.25% | 0.188729 |
| punctuation | 5014 | 1538 | 10.03% | 10.72% | 6.94% | 0.007935 |
| subordinating conjunction | 55 | 16 | 0.11% | 0.11% | 5.79% | 0.439577 |
| symbol | 311 | 66 | 0.62% | 0.46% | -25.17% | 0.012121 |
| verb | 6889 | 1999 | 13.78% | 13.94% | 1.15% | 0.314955 |

### English (Waseem)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 7100 | 3817 | 7.04% | 8.92% | 26.69% | 0.000000 |
| adposition | 3447 | 1797 | 3.42% | 4.20% | 22.87% | 0.000000 |
| adverb | 3799 | 1462 | 3.77% | 3.41% | -9.29% | 0.000571 |
| auxiliary | 2902 | 987 | 2.88% | 2.31% | -19.81% | 0.000000 |
| coordinating conjunction | 575 | 136 | 0.57% | 0.32% | -43.96% | 0.000000 |
| determiner | 541 | 245 | 0.54% | 0.57% | 6.94% | 0.194330 |
| interjection | 1203 | 389 | 1.19% | 0.91% | -23.68% | 0.000001 |
| noun | 25150 | 10748 | 24.93% | 25.10% | 0.70% | 0.242575 |
| numeral | 2319 | 764 | 2.30% | 1.78% | -22.31% | 0.000000 |
| particle | 2065 | 1038 | 2.05% | 2.42% | 18.50% | 0.000004 |
| pronoun | 842 | 377 | 0.83% | 0.88% | 5.65% | 0.189135 |
| proper noun | 9271 | 4269 | 9.19% | 9.97% | 8.51% | 0.000002 |
| punctuation | 23645 | 9442 | 23.44% | 22.05% | -5.90% | 0.000000 |
| subordinating conjunction | 113 | 54 | 0.11% | 0.13% | 13.68% | 0.225120 |
| symbol | 3817 | 1209 | 3.78% | 2.82% | -25.33% | 0.000000 |
| verb | 13277 | 5775 | 13.16% | 13.49% | 2.50% | 0.046840 |

### French (Ousidhoum)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 1455 | 410 | 12.15% | 14.74% | 21.45% | 0.000128 |
| adposition | 1019 | 174 | 8.51% | 6.25% | -26.18% | 0.000030 |
| adverb | 809 | 173 | 6.76% | 6.22% | -7.58% | 0.158707 |
| auxiliary | 448 | 104 | 3.74% | 3.74% | 0.61% | 0.485511 |
| coordinating conjunction | 46 | 14 | 0.38% | 0.50% | 37.31% | 0.162021 |
| determiner | 397 | 106 | 3.31% | 3.81% | 15.67% | 0.093206 |
| noun | 2031 | 574 | 16.96% | 20.63% | 21.75% | 0.000003 |
| numeral | 112 | 24 | 0.94% | 0.86% | -4.81% | 0.393083 |
| pronoun | 804 | 153 | 6.71% | 5.50% | -17.69% | 0.009285 |
| proper noun | 195 | 56 | 1.63% | 2.01% | 25.12% | 0.074480 |
| punctuation | 963 | 233 | 8.04% | 8.38% | 4.44% | 0.272276 |
| subordinating conjunction | 143 | 25 | 1.19% | 0.90% | -22.32% | 0.101560 |
| verb | 3163 | 629 | 26.41% | 22.61% | -14.33% | 0.000015 |

### German (Bretschneider)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 6505 | 2096 | 7.64% | 8.64% | 13.21% | 0.000000 |
| adposition | 1036 | 388 | 1.22% | 1.60% | 31.75% | 0.000003 |
| adverb | 12361 | 3480 | 14.51% | 14.35% | -1.10% | 0.265836 |
| auxiliary | 1016 | 321 | 1.19% | 1.32% | 11.21% | 0.050016 |
| coordinating conjunction | 215 | 69 | 0.25% | 0.28% | 13.82% | 0.181943 |
| determiner | 720 | 248 | 0.85% | 1.02% | 21.30% | 0.004879 |
| interjection | 19 | 7 | 0.02% | 0.03% | 40.49% | 0.234413 |
| noun | 16513 | 4801 | 19.39% | 19.80% | 2.13% | 0.076430 |
| numeral | 1260 | 250 | 1.48% | 1.03% | -30.09% | 0.000000 |
| particle | 179 | 58 | 0.21% | 0.24% | 15.12% | 0.183456 |
| pronoun | 967 | 225 | 1.14% | 0.93% | -18.00% | 0.002805 |
| proper noun | 6093 | 1766 | 7.15% | 7.28% | 1.84% | 0.243388 |
| punctuation | 22835 | 6234 | 26.81% | 25.71% | -4.10% | 0.000300 |
| subordinating conjunction | 236 | 72 | 0.28% | 0.30% | 8.18% | 0.287615 |
| verb | 13534 | 3867 | 15.89% | 15.95% | 0.37% | 0.412822 |

### German (Ross)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 515 | 160 | 8.11% | 9.06% | 12.10% | 0.097576 |
| adposition | 62 | 15 | 0.98% | 0.85% | -8.76% | 0.349951 |
| adverb | 504 | 146 | 7.94% | 8.27% | 4.58% | 0.315117 |
| auxiliary | 43 | 16 | 0.68% | 0.91% | 38.81% | 0.140680 |
| coordinating conjunction | 29 | 5 | 0.46% | 0.28% | -28.14% | 0.194892 |
| determiner | 42 | 13 | 0.66% | 0.74% | 16.97% | 0.326273 |
| noun | 1147 | 308 | 18.06% | 17.44% | -3.30% | 0.278446 |
| numeral | 86 | 12 | 1.35% | 0.68% | -46.32% | 0.009857 |
| particle | 8 | 1 | 0.13% | 0.06% | -20.16% | 0.324685 |
| pronoun | 48 | 12 | 0.76% | 0.68% | -4.68% | 0.415218 |
| proper noun | 1556 | 389 | 24.50% | 22.03% | -10.01% | 0.015445 |
| punctuation | 1047 | 301 | 16.49% | 17.04% | 3.53% | 0.283749 |
| verb | 849 | 244 | 13.37% | 13.82% | 3.56% | 0.306374 |

### Indonesian (Alfina)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 141 | 72 | 2.53% | 1.74% | -30.61% | 0.004382 |
| adposition | 116 | 87 | 2.08% | 2.10% | 1.53% | 0.459385 |
| adverb | 18 | 23 | 0.32% | 0.56% | 70.51% | 0.039953 |
| auxiliary | 7 | 5 | 0.13% | 0.12% | 1.24% | 0.498601 |
| coordinating conjunction | 14 | 26 | 0.25% | 0.63% | 142.97% | 0.002177 |
| determiner | 10 | 12 | 0.18% | 0.29% | 59.53% | 0.126439 |
| interjection | 10 | 10 | 0.18% | 0.24% | 34.99% | 0.243467 |
| noun | 609 | 382 | 10.91% | 9.24% | -15.25% | 0.003511 |
| numeral | 77 | 21 | 1.38% | 0.51% | -61.93% | 0.000006 |
| particle | 1 | 2 | 0.02% | 0.05% | 102.48% | 0.209817 |
| pronoun | 21 | 8 | 0.38% | 0.19% | -44.78% | 0.054872 |
| proper noun | 3132 | 2492 | 56.12% | 60.28% | 7.41% | 0.000020 |
| punctuation | 998 | 742 | 17.88% | 17.95% | 0.39% | 0.464920 |
| symbol | 145 | 69 | 2.60% | 1.67% | -35.28% | 0.000946 |
| verb | 245 | 146 | 4.39% | 3.53% | -19.34% | 0.016663 |

### Indonesian (Ibrohim)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 2972 | 1422 | 2.19% | 1.62% | -26.02% | 0.000000 |
| adposition | 700 | 393 | 0.52% | 0.45% | -13.13% | 0.011890 |
| adverb | 654 | 364 | 0.48% | 0.42% | -13.87% | 0.010365 |
| auxiliary | 125 | 43 | 0.09% | 0.05% | -46.03% | 0.000106 |
| coordinating conjunction | 445 | 328 | 0.33% | 0.37% | 14.01% | 0.036213 |
| determiner | 423 | 97 | 0.31% | 0.11% | -64.28% | 0.000000 |
| interjection | 424 | 298 | 0.31% | 0.34% | 8.73% | 0.134918 |
| noun | 17792 | 11206 | 13.13% | 12.78% | -2.65% | 0.008339 |
| numeral | 2176 | 1069 | 1.61% | 1.22% | -24.04% | 0.000000 |
| particle | 1702 | 1079 | 1.26% | 1.23% | -1.99% | 0.300793 |
| pronoun | 369 | 156 | 0.27% | 0.18% | -34.42% | 0.000003 |
| proper noun | 72498 | 51335 | 53.49% | 58.54% | 9.44% | 0.000000 |
| punctuation | 27889 | 16088 | 20.58% | 18.34% | -10.84% | 0.000000 |
| symbol | 1624 | 650 | 1.20% | 0.74% | -38.08% | 0.000000 |
| verb | 4598 | 2718 | 3.39% | 3.10% | -8.63% | 0.000070 |

### Italian (Bosco)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 14600 | 11925 | 18.91% | 19.93% | 5.40% | 0.000001 |
| adposition | 1918 | 1181 | 2.48% | 1.97% | -20.52% | 0.000000 |
| adverb | 3150 | 3111 | 4.08% | 5.20% | 27.44% | 0.000000 |
| auxiliary | 2732 | 2362 | 3.54% | 3.95% | 11.57% | 0.000037 |
| coordinating conjunction | 169 | 137 | 0.22% | 0.23% | 4.75% | 0.344313 |
| determiner | 1803 | 1441 | 2.33% | 2.41% | 3.14% | 0.188016 |
| interjection | 61 | 59 | 0.08% | 0.10% | 24.88% | 0.110901 |
| noun | 19385 | 14556 | 25.10% | 24.32% | -3.10% | 0.000454 |
| numeral | 2521 | 1274 | 3.26% | 2.13% | -34.76% | 0.000000 |
| pronoun | 766 | 687 | 0.99% | 1.15% | 15.75% | 0.002606 |
| proper noun | 5923 | 3238 | 7.67% | 5.41% | -29.45% | 0.000000 |
| punctuation | 11668 | 9240 | 15.11% | 15.44% | 2.19% | 0.045705 |
| subordinating conjunction | 160 | 166 | 0.21% | 0.28% | 33.85% | 0.004158 |
| symbol | 153 | 147 | 0.20% | 0.25% | 24.01% | 0.031010 |
| verb | 10421 | 9349 | 13.49% | 15.62% | 15.77% | 0.000000 |

### Italian (Manuel)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 8944 | 1829 | 9.05% | 9.95% | 9.89% | 0.000068 |
| adposition | 8845 | 1502 | 8.95% | 8.17% | -8.74% | 0.000258 |
| adverb | 3695 | 980 | 3.74% | 5.33% | 42.56% | 0.000000 |
| auxiliary | 2573 | 657 | 2.60% | 3.57% | 37.31% | 0.000000 |
| coordinating conjunction | 2267 | 562 | 2.29% | 3.06% | 33.33% | 0.000000 |
| determiner | 5813 | 1320 | 5.88% | 7.18% | 22.04% | 0.000000 |
| interjection | 164 | 12 | 0.17% | 0.07% | -57.68% | 0.000276 |
| noun | 20122 | 3680 | 20.37% | 20.01% | -1.75% | 0.134489 |
| numeral | 1884 | 230 | 1.91% | 1.25% | -34.18% | 0.000000 |
| pronoun | 3357 | 1064 | 3.40% | 5.79% | 70.35% | 0.000000 |
| proper noun | 12387 | 1397 | 12.54% | 7.60% | -39.38% | 0.000000 |
| punctuation | 16025 | 2378 | 16.22% | 12.93% | -20.27% | 0.000000 |
| subordinating conjunction | 729 | 198 | 0.74% | 1.08% | 46.42% | 0.000002 |
| symbol | 140 | 36 | 0.14% | 0.20% | 40.95% | 0.039382 |
| verb | 10287 | 2360 | 10.41% | 12.83% | 23.27% | 0.000000 |

### Portuguese
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 9679 | 2111 | 13.77% | 11.73% | -14.78% | 0.000000 |
| adposition | 1920 | 492 | 2.73% | 2.73% | 0.24% | 0.484755 |
| adverb | 2744 | 804 | 3.90% | 4.47% | 14.54% | 0.000325 |
| auxiliary | 1201 | 417 | 1.71% | 2.32% | 35.83% | 0.000000 |
| coordinating conjunction | 82 | 28 | 0.12% | 0.16% | 36.47% | 0.085803 |
| determiner | 1805 | 482 | 2.57% | 2.68% | 4.46% | 0.197767 |
| interjection | 136 | 37 | 0.19% | 0.21% | 8.34% | 0.345082 |
| noun | 17602 | 4618 | 25.04% | 25.66% | 2.49% | 0.043248 |
| numeral | 1557 | 491 | 2.21% | 2.73% | 23.34% | 0.000029 |
| pronoun | 852 | 318 | 1.21% | 1.77% | 46.07% | 0.000000 |
| proper noun | 7245 | 1399 | 10.31% | 7.77% | -24.54% | 0.000000 |
| punctuation | 13353 | 2919 | 18.99% | 16.22% | -14.59% | 0.000000 |
| subordinating conjunction | 500 | 189 | 0.71% | 1.05% | 48.13% | 0.000004 |
| symbol | 38 | 7 | 0.05% | 0.04% | -19.88% | 0.252510 |
| verb | 11267 | 3612 | 16.03% | 20.07% | 25.24% | 0.000000 |

### Spanish (Basile)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 11100 | 7463 | 16.45% | 16.96% | 3.13% | 0.012196 |
| adposition | 292 | 190 | 0.43% | 0.43% | -0.02% | 0.496719 |
| adverb | 1565 | 1058 | 2.32% | 2.40% | 3.72% | 0.177282 |
| auxiliary | 899 | 565 | 1.33% | 1.28% | -3.54% | 0.247787 |
| coordinating conjunction | 326 | 188 | 0.48% | 0.43% | -11.35% | 0.090821 |
| determiner | 572 | 405 | 0.85% | 0.92% | 8.68% | 0.100069 |
| interjection | 96 | 52 | 0.14% | 0.12% | -16.20% | 0.144959 |
| noun | 16375 | 10185 | 24.27% | 23.15% | -4.60% | 0.000009 |
| numeral | 1628 | 892 | 2.41% | 2.03% | -15.92% | 0.000011 |
| pronoun | 399 | 274 | 0.59% | 0.62% | 5.45% | 0.250463 |
| proper noun | 9135 | 5924 | 13.54% | 13.46% | -0.53% | 0.365850 |
| punctuation | 11438 | 7720 | 16.95% | 17.55% | 3.53% | 0.004905 |
| subordinating conjunction | 580 | 436 | 0.86% | 0.99% | 15.36% | 0.012050 |
| symbol | 2967 | 1974 | 4.40% | 4.49% | 2.06% | 0.236701 |
| verb | 9826 | 6501 | 14.56% | 14.78% | 1.48% | 0.159855 |
| space | 285 | 169 | 0.42% | 0.38% | -8.83% | 0.166559 |

### Spanish (Pereira)
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 11158 | 3924 | 16.80% | 18.45% | 9.83% | 0.000000 |
| adposition | 334 | 62 | 0.50% | 0.29% | -41.28% | 0.000017 |
| adverb | 1641 | 485 | 2.47% | 2.28% | -7.58% | 0.058753 |
| auxiliary | 932 | 355 | 1.40% | 1.67% | 19.14% | 0.002674 |
| coordinating conjunction | 250 | 83 | 0.38% | 0.39% | 4.50% | 0.371782 |
| determiner | 544 | 184 | 0.82% | 0.87% | 5.99% | 0.251655 |
| interjection | 93 | 26 | 0.14% | 0.12% | -10.31% | 0.293615 |
| noun | 14591 | 4707 | 21.97% | 22.13% | 0.74% | 0.309234 |
| numeral | 1593 | 386 | 2.40% | 1.82% | -24.19% | 0.000000 |
| particle | 3 | 2 | 0.00% | 0.01% | 134.18% | 0.157848 |
| pronoun | 378 | 117 | 0.57% | 0.55% | -2.78% | 0.386884 |
| proper noun | 9332 | 2623 | 14.05% | 12.33% | -12.21% | 0.000000 |
| punctuation | 12655 | 3856 | 19.06% | 18.13% | -4.84% | 0.001317 |
| subordinating conjunction | 720 | 192 | 1.08% | 0.90% | -16.42% | 0.011401 |
| symbol | 2962 | 1286 | 4.46% | 6.05% | 35.63% | 0.000000 |
| verb | 8939 | 2942 | 13.46% | 13.83% | 2.79% | 0.082795 |
| space | 284 | 37 | 0.43% | 0.17% | -58.37% | 0.000000 |

### Turkish
| Part of Speech (POS) |  Normal Speech (NS) Count  | Hate Speech (HS) Count | POS % in NS | POS % in HS | Lift | p-value |
|:-----|:--------:|:------:|:------:|:------:|:------:|:------:|
| adjective | 17102 | 5097 | 3.79% | 3.74% | -1.06% | 0.246103 |
| adposition | 19418 | 5540 | 4.30% | 4.07% | -5.29% | 0.000122 |
| adverb | 3317 | 901 | 0.73% | 0.66% | -9.77% | 0.002710 |
| auxiliary | 994 | 264 | 0.22% | 0.19% | -11.60% | 0.034409 |
| coordinating conjunction | 395 | 102 | 0.09% | 0.07% | -13.67% | 0.085475 |
| determiner | 962 | 232 | 0.21% | 0.17% | -19.69% | 0.001010 |
| interjection | 603 | 181 | 0.13% | 0.13% | 0.01% | 0.494634 |
| noun | 73292 | 20610 | 16.22% | 15.14% | -6.66% | 0.000000 |
| numeral | 6264 | 1850 | 1.39% | 1.36% | -1.94% | 0.226648 |
| particle | 479 | 285 | 0.11% | 0.21% | 97.77% | 0.000000 |
| pronoun | 705 | 196 | 0.16% | 0.14% | -7.38% | 0.164957 |
| proper noun | 255988 | 79945 | 56.67% | 58.74% | 3.66% | 0.000000 |
| punctuation | 47804 | 13664 | 10.58% | 10.04% | -5.12% | 0.000000 |
| subordinating conjunction | 198 | 75 | 0.04% | 0.06% | 26.76% | 0.044088 |
| symbol | 3218 | 1075 | 0.71% | 0.79% | 10.95% | 0.001698 |
| verb | 17346 | 5183 | 3.84% | 3.81% | -0.81% | 0.299049 |

## Sentiment Statistics (sentiment.ipynb)

Here we evaluate if there are any significant differences in sentiment between the hate speech and normal examples. In order to evaluate all non-English sentiment's the original text is first translated to English. 

Methodology: Concat entire dataset, take vader score.

| Dataset |  Hate Speech Negative Sentiment  | Normal Speech Negative Sentiment |
|:-----|:--------:|:------:|
| arabic_mulki   | 0.067 | 0.06 |
| danish | 0.193 | 00.1 |
| english_basile | 0.361 | 0.222 |
| english_davidson | 0.359 | 0.085 |
| english_founta | 0.297 | 0.111 |
| english_gilbert | 0.186 | 0.118 |
| english_ousidhoum | 0.317 | 0.291 |
| english_waseem | 0.131 | 0.101 |
| french_ousidhoum | 0.156 | 0.143 |
| german_bertschneider | 0.164 | 0.128 |
| german_ross | 0.191 | 0.148 |
| indonesian_alfina | 0.14 | 0.059 |
| indonesian_ibrohim | 0 | 0 |
| italian_bosco | 0 | 0 |
| italian_manuel | 0.188 | 0.169 |
| portuguese | 0 | 0 |
| spanish_basile | 0 | 0 |
| spanish_pereira | 0 | 0 |
| turkish | 0 | 0 |
