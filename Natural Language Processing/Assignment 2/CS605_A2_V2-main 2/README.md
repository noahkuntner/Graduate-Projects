# Setup Instructions [IMPORTANT]

Run the following in command line to setup the virtual environment

1. `pip install -r requirements.txt`
2. `python -m spacy download en_core_web_sm`

Next, open up a python interpreter and run the following to set up `nltk`:

1. `nltk.download('stopwords')`
2. `nltk.download('wordnet')`

Packages specified in `requirements.txt` are:

|package|version|comments|
|---|---|---|
|`torch`|`1.7.0+cu101`|Use `pip` in anaconda|
|`torchtext`|`0.8.0`||
|`transformers`|`4.5.0`||
|`numpy`|`1.19.5`||
|`spacy`|`3.0.6`|Install `en_core_web_sm`|
|`nltk`|`3.6.2`||

# Layout

## Folder structure

The project consists of the following folders:

- `src`:        Stores python modules containing code for pipeline such as preprocessing, model specifications, prediction etc.
- `data`:       Contains all data files, including preprocessed train data. Please place unseen data for prediction here

The code will create the following folders:

- `training_results`:   Stores the results of the training for each model
- `charts`:             Visualizations for the training process
- `model`:              Saves the models here
- `vocab`:              Saves the generated Vocab object used for prediction
- `predictions`:        Saves `.csv` containing *pre-processed* prediction text and sentiment classification (`0` for negative or `1` for positive)

## YAML parameters

The repo contains a `parameters.yml` file that contains all critical parameters and values.

Change important experiment parameters in this file, including:
- model architecture
- file directories
- prebuilt word vectors and more.

**IMPORTANT**: Change `model_name` and `model_dict_name` on each training run to avoid overwriting other experiments.

----

# Directions

## 1. Preprocessing
This is straight forward - run `preprocess.py`.

## 2. Training the model
This is straight forward - run `run.py`.

Parameters will be extracted from `parameters.yml`, and the relevant folders created for output.

## 3. Prediction
This file is run from the command line, and the required syntax is:

`python predict.py <model_name> <target_file>`

The script will:
- Look for `model_name` in the directory specified in the `YAML` parameter `model_parameters`>`model_dir`
- Look for data in the `data` folder
- Preprocess the text

After loading the saved `Vocab` object and model specified in `parameters.yml`, the model will be used to predict the sentiment of the input. If the score is above the threshold set in parameters (default is 0.5), it will return 1 (positive sentiment), else it will return 0 (negative sentiment).

The results will be written to `results.csv` under the `results` folder with the format

|index|text_preprocessed|prediction|
|---|---|---|
|1|preprocessed text 1|predicted sentiment|
|2|preprocessed text 2|predicted sentiment|

...and so on.

----
# Description of Model

The network is constructed using `pytorch 1.7` and `torchtext 0.8.1`, and consists of the following layers and hyperparameters:

1. `nn.Embedding()`
    - `vocab_size` - taken from `parameters.yml`
    - `embedding_dim`- taken from `parameters.yml`
    - `pad_idx` - taken from `parameters.yml`
2. `nn.LSTM()`
    - `embedding_dim` - taken from `parameters.yml`
    - `hidden_dim` - taken from `parameters.yml`
    - `n_layers` - taken from `parameters.yml`
    - `bidrectional` - taken from `parameters.yml`
3. `nn.Linear`
4. `nn.Dropout`
    - `dropout` - taken from `parameters.yml`

## Optimizer

The optimizer used was Adam (<u>**Ada**</u>ptive <u>**M**</u>oment Estimation), an extension to the commonly used SGD. Where SGD only uses a single learning rate, Adam maintains a learning rate for each parameter, which improves performance on problems with sparse gradients such as NLP.

## Loss

The loss criterion chosen was `BCEWithLogitsLoss()`, which is a binary crossentropy loss bundled with a sigmoid layer.