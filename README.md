# Highly Generalizable Models for Multilingual Hate Speech Detection

## Abstract

Hate speech detection has become an important research topic within the past decade. More private corporations are needing to regulate user generated content on different platforms across the globe. In this paper, we introduce a study of multilingual hate speech classification. We compile a dataset of 11 languages and resolve different taxonomies by analyzing the combined data with binary labels: hate speech or not hate speech. Defining hate speech in a single way across different languages and datasets may erase cultural nuances to the definition, therefore, we utilize language agnostic embeddings provided by LASER and MUSE in order to develop models that can use a generalized definition of hate speech across datasets. Furthermore, we evaluate prior state of the art methodologies for hate speech detection under our expanded dataset. We conduct three types of experiments for a binary hate speech classification task: Multilingual-Train Monolingual-Test,  Monolingual-Train Monolingual-Test and Language-Family-Train Monolingual Test scenarios to see if performance increases for each language due to learning more from other language data. 

## Set up the environment
We use an anaconda environment for all the libraries we use in our experiments. The environment has been saved within the `env.txt` file. To create the environment, run the following command:

`conda env create --file env.txt`
## How this repo is structured
The root of the repo consists of the following files/directories:
1. `data`: This folder contains all variations of our datasets.
2. `models`: This folder contains implementations of the models we train to recognize hate speech and the notebooks that run experiments using those models.
3. `preprocessing`: This folder contains all the notebooks that perform rudimentary statistical analysis and preprocessing for all the datasets we use.
