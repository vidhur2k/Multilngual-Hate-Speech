# Dataset Preprocessing and Statistics

The following files are used to process the input data so that statistics may be performed prior to the implementation of black box machine learning models. 
The purpose of these files is to provide the developer domain insights as to how hate speech differs from non-hate speech and how hate speech compares across multiple languages.

## Word Count Statistics (wordcount.ipynb)

This portion of the project begins with a simple word count analysis comparing hate speech versus non hate speech for each language. If the language specific dataset offers additional taxonomies with labelings then statistics will be performed on these labelings as well to improve understanding of the taxonomy definitions. We continue this work by examining if there are statistically significant differences in the usage of parts of speech between the hate speech and normal examples.

## Sentiment Statistics (sentiment.ipynb)

Here we evaluate if there are any significant differences in sentiment between the hate speech and normal examples. In order to evaluate all non-English sentiment's the original text is first translated to English. 

| Dataset |  Hate Speech Negative Sentiment  | Normal Speech Negative Sentiment |
|:-----|:--------:|------:|
| arabic_mulki   | 0 | 0 |
| danish | 0 | 0 |
| english_basile | .361 | .222 |
| english_davidson | .359 | .085 |
| english_founta | .297 | .111 |
| english_gilbert | .186 | .118 |
| english_ousidhoum | .317 | .291 |
| english_waseem | .131 | .101 |
| french_ousidhoum | 0 | 0 |
| german_bertschneider | 0 | 0 |
| german_ross | 0 | 0 |
| indonesian_alfina | 0 | 0 |
| indonesian_ibrohim | 0 | 0 |
| italian_bosco | 0 | 0 |
| italian_manuel | 0 | 0 |
| portuguese | 0 | 0 |
| spanish_basile | 0 | 0 |
| spanish_pereira | 0 | 0 |
| turkish | 0 | 0 |
