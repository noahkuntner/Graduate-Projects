import torch.nn as nn
import torch


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        """Initializes our LSTM network.
        Args:
            vocab_size (int):       for nn.Embedding which holds Tensor of dimension (vocab_size,embedding_dim)
            embedding_dim (int):    for nn.Embedding which holds Tensor of dimension (vocab_size,embedding_dim)
            hidden_dim (int):       number of features in the hidden state h for nn.LSTM
            output_dim (int):       number of features in final fully connected layer
            n_layers (int):         number of layers in LSTM
            bidirectional (bool):   toggles bidirectionality on or off
            dropout (float):        float between 0 and 1. Determines dropout probability between each LSTM layer
            pad_idx (int):          freezes embedding vector at padding_idx during training, i.e. it remains as a fixed “pad”
        
        Outputs:
            self.fc

        """

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        # lengths need to be on CPU!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.to('cpu'))

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict_sentiment(model, sentence, TEXT, nlp, device):
    """Returns the sentiment score of a string
    Args:
        model (RNN class):      Our LSTM model
        sentence (str):         Sentence to be analyzed
        TEXT (torchtext.Vocab): Vocab from corpus
        nlp (spacy tokenizer):  spacy tokenizer
        device (torch.device):  Device to predict on
        
    Yields:
        result (float):         Sentiment score for the sentence
    """
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    result = prediction.item()   
    print(f"Sentiment = [{result:.3f}] for {sentence[:50]}...")
    return result
