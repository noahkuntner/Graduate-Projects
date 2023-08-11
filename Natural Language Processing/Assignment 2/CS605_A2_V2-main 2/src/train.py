import torch
import os


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    """Trains the model.

    Args:
        model (RNN class object):   RNN Model to be trained
        iterator (object):          Dataset iterator
        optimizer (object):         optim.Adam()
        criterion (object):         BCEWithLogitsLoss()

    Yields:
        epoch_loss (float):         Epoch loss
        epoch_acc (float):          Epoch accuracy

    Notes:
        Updates weights for model in place.
    """

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:

        optimizer.zero_grad()

        text, text_lengths = batch.Text_preprocessed

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.Label)

        acc = binary_accuracy(predictions, batch.Label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """Computes the validation test scores for epoch
    
    Args:
        model (RNN class object):   Model to be trained
        iterator (object):          Data iterator to be evaluate model on
        criterion (object):         BCEWithLogitsLoss()
    Yields
        epoch_loss (float):         Epoch loss
        epoch_acc (float):          Epoch accuracy
    """
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            text, text_lengths = batch.Text_preprocessed

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.Label)

            acc = binary_accuracy(predictions, batch.Label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """Calculates time for the epoch"""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_results(filename, model_name, loss, acc):
    """Checks if results file exists, and creates it or appends results to it"""
    if os.path.exists(filename):
        append_write = 'a+'
    else:
        append_write = 'w+'

    with open(filename, append_write, encoding="utf-8") as f:
        if append_write == 'w+':
            f.write(f"model_name,loss,acc\n")
        f.write(f"{model_name},{loss:.4f},{acc:.4f}\n")
