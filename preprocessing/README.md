# Dataset Preprocessing and Statistics

The following files are used to process the input data so that statistics may be performed prior to the implementation of black box machine learning models. 
The purpose of these files is to provide the developer domain insights as to how hate speech differs from non-hate speech and how hate speech compares across multiple languages.

## Word Count Statistics (wordcount.ipynb)

This portion of the project begins with a simple word count analysis comparing hate speech versus non hate speech for each language. If the language specific dataset offers additional taxonomies with labelings then statistics will be performed on these labelings as well to improve understanding of the taxonomy definitions. We continue this work by examining if there are statistically significant differences in the usage of parts of speech between the hate speech and normal examples.

## Sentiment Statistics (sentiment.ipynb)

Here we evaluate if there are any significant differences in sentiment between the hate speech and normal examples. In order to evaluate all non-English sentiment's the original text is first translated to English. 