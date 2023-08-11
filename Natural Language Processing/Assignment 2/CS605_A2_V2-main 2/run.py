import nltk
import torch
import torch.optim as optim
import torchtext
from torchtext import data
import spacy

import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

import sys
import pickle
import time
import os
import warnings

from src.preprocessing import *
from src.model import *
from src.train import *

print(f"System version: {sys.version}")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"Torchtext version: {torchtext.__version__}")
print(f"Spacy version: {spacy.__version__}")
print(f"Spacy version: {nltk.__version__}")

warnings.filterwarnings("ignore")

parameters = load_parameters('parameters.yml')
#%% Data loading
train_path = parameters['preprocessing_parameters']['train_path']
test_path = parameters['preprocessing_parameters']['test_path']
vectors = parameters['preprocessing_parameters']['vectors']
tokenizer = parameters['preprocessing_parameters']['tokenizer']
tokenizer_language = parameters['preprocessing_parameters']['tokenizer_language']
data_path = parameters['preprocessing_parameters']['data_path']

SEED = parameters['preprocessing_parameters']['SEED']
MAX_VOCAB_SIZE = parameters['preprocessing_parameters']['MAX_VOCAB_SIZE']
BATCH_SIZE = parameters['preprocessing_parameters']['BATCH_SIZE']

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = parameters['preprocessing_parameters']['deterministic']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEXT = data.Field(tokenize=tokenizer,
                  tokenizer_language=tokenizer_language,
                  include_lengths=True)

LABEL = data.LabelField(dtype=torch.float)

train_data, valid_data, test_data = split_data(
    f"preprocessed_{train_path}", f"preprocessed_{test_path}", TEXT, LABEL, random.seed(SEED), data_path)

print("Building vocabulary...")
build_vocab(TEXT=TEXT, LABEL=LABEL, train_data=train_data,
            MAX_VOCAB_SIZE=MAX_VOCAB_SIZE, vectors=vectors)
print("Done!")

print(f"Generating dataset iterators...")
train_iterator, valid_iterator, test_iterator = get_iterators(
    train_data, valid_data, test_data, BATCH_SIZE=BATCH_SIZE)
print("Done!")

filename = parameters['folders']['vocab_filename']
vocab_dir = parameters['folders']['vocab_dir']

if not os.path.exists(vocab_dir):
    os.makedirs(vocab_dir)

with open(f"{vocab_dir}\{filename}", 'wb') as f:
    pickle.dump(TEXT, f)

print(f"Saved Vocab object [{filename}] at [{vocab_dir}] for later retrieval!")

#%% Model construction

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = parameters['model_parameters']['EMBEDDING_DIM']
HIDDEN_DIM = parameters['model_parameters']['HIDDEN_DIM']
OUTPUT_DIM = parameters['model_parameters']['OUTPUT_DIM']
N_LAYERS = parameters['model_parameters']['N_LAYERS']
BIDIRECTIONAL = parameters['model_parameters']['BIDIRECTIONAL']
DROPOUT = parameters['model_parameters']['DROPOUT']
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = RNN(INPUT_DIM,
            EMBEDDING_DIM,
            HIDDEN_DIM,
            OUTPUT_DIM,
            N_LAYERS,
            BIDIRECTIONAL,
            DROPOUT,
            PAD_IDX)


print(f'The model has {count_parameters(model):,} trainable parameters')

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)

model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

#%% Model training

N_EPOCHS = parameters['model_parameters']['N_EPOCHS']
model_dict_name = parameters['folders']['model_dict_name']
best_valid_loss = float('inf')

train_history = pd.DataFrame({'train_loss': [], 'train_acc': []})
valid_history = pd.DataFrame({'valid_loss': [], 'valid_acc': []})

model_dir = parameters['folders']['model_dir']

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f"Saving model parameters to [{model_dir}/{model_dict_name}]")
        torch.save(model.state_dict(), f"{model_dir}/{model_dict_name}")

    train_result_dict = {'train_loss': train_loss, 'train_acc': train_acc}
    valid_result_dict = {'valid_loss': valid_loss, 'valid_acc': valid_acc}

    train_history.loc[epoch] = train_result_dict
    valid_history.loc[epoch] = valid_result_dict

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

#%% Data visualization

chart_dir = parameters['folders']['chart_dir']

if not os.path.exists(chart_dir):
    os.makedirs(chart_dir)

train_hist = train_history.plot(title=f"Training History - {model_dict_name}")
plt.savefig(f"{chart_dir}/{model_dict_name}_train_hist.png")

valid_hist = valid_history.plot(
    title=f"Validation History - {model_dict_name}")
plt.savefig(f"{chart_dir}/{model_dict_name}_val_hist.png")
print(f"Charts saved at [{chart_dir}]!")


#%% Model testing
model.load_state_dict(torch.load(f"{model_dir}\{model_dict_name}"))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

model_name = parameters['folders']['model_name']

print(f"Saving model as [{model_name}]")
torch.save(model, f"{model_dir}/{model_name}")

results_folder = parameters['folders']['results']
results_file = parameters['folders']['results_filename']

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

save_results(f"{results_folder}/{results_file}", model_name, test_loss, test_acc)
print(f"Results saved in [results] folder")