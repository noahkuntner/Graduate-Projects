import spacy
import torch
from src.preprocessing import *
from src.model import *
import sys
import os
import re
import pandas as pd


def parse_file(file_name):
    """Cleans text 
    """

    def parse_review(string):
        # exp = r"review \d+ (.*)"
        exp = r"review (\d+) (.*\b)"
        a = re.compile(exp)
        result = a.search(string)
        return result[1], result[2]

    with open(f"data\{file_name}", "r", encoding='utf-8-sig') as f:
        my_list = f.read().splitlines()

    data = pd.DataFrame(list(map(parse_review, my_list)), columns=[
                        'index', 'text']).set_index('index')
    data['text_preprocessed'] = data['text'].apply(clean_text)

    return data['text_preprocessed']


if __name__ == '__main__':
    # To use this script, use "python predict.py model_name target_file" in command line
    assert len(sys.argv) == 3, "Please use the following syntax: python predict.py model_name target_file"
    parameters = load_parameters('parameters.yml')

    model_name = sys.argv[1]
    target_file = sys.argv[2]

    vocab_filename = parameters['folders']['vocab_filename']
    vocab_dir = parameters['folders']['vocab_dir']

    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    TEXT_load = load_vocab(f"{vocab_dir}\{vocab_filename}")

    tokenizer_language = parameters['preprocessing_parameters']['tokenizer_language']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nlp = spacy.load(tokenizer_language)

    model_dir = parameters['folders']['model_dir']
    model = torch.load(f"{model_dir}\{model_name}")
    model.eval()

    result = parse_file(target_file)
    sentiment = result.apply(lambda x: predict_sentiment(model, x, TEXT_load, nlp, device))
    threshold = parameters['prediction_parameters']['sentiment_threshold']
    sentiment = sentiment.apply(lambda x: 1 if x > threshold else 0)
    result = pd.concat([result, sentiment.rename('prediction')], axis=1)

    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    result.to_csv(f"predictions\{model_name}.csv")
