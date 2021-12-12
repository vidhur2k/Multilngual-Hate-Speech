# Highly Generalizable Models for Multilingual Hate Speech Detection

## Abstract

## Set up the environment
We use an anaconda environment for all the libraries we use in our experiments. The environment has been saved within the `env.txt` file. To create the environment, run the following command:

`conda env create --file env.txt`
## How this repo is structured
The root of the repo consists of the following files/directories:
1. `data`: This folder contains all variations of our datasets.
2. `models`: This folder contains implementations of the models we train to recognize hate speech and the notebooks that run experiments using those models.
3. `preprocessing`: This folder contains all the notebooks that perform rudimentary statistical analysis and preprocessing for all the datasets we use.