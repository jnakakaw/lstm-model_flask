from flask import Flask, request, Response, json
import numpy as np
import random
import os

import torch
torch.use_deterministic_algorithms(True)  # to help make code deterministic
torch.backends.cudnn.benchmark = False  # to help make code deterministic
import torch.nn as nn

np.random.seed(0)  # to help make code deterministic
torch.manual_seed(0)  # to help make code deterministic
random.seed(0)  # to help make code deterministic

import pickle

with open('./lstm/vocab.pkl', 'rb') as input:
    vocab = pickle.load(input)

with open('./lstm/embedding_matrix.pkl', 'rb') as input:
    embedding_matrix = pickle.load(input)

with open('./lstm/tokenizer.pkl', 'rb') as input:
    tokenizer = pickle.load(input)

# flask instance
app = Flask(__name__)

# Create API
@app.route("/api", methods=["GET", "POST"])
def predict():
    # get data from request
    data = request.get_json(force=True)

    Input_Text = float(data["Input_Text"])

    if Input_Text == "":
        return ""

    def tokenizer_lowercase(text):
        return [token.lower() for token in tokenizer(text)]

    class EmbeddingLSTMLinearModel(nn.Module):
        def __init__(self, embedding_matrix, num_lstm_output_nodes, num_final_output_nodes):
            super().__init__()
            self.embedding_layer = nn.Embedding.from_pretrained(embedding_matrix)
            self.lstm_layer = nn.LSTM(embedding_matrix.shape[1], num_lstm_output_nodes)
            self.linear_layer = nn.Linear(num_lstm_output_nodes, num_final_output_nodes)

        def forward(self, text_encodings, lengths):
            embeddings = self.embedding_layer(text_encodings)

            rnn_last_time_step_outputs = \
                UDA_get_rnn_last_time_step_outputs(embeddings, lengths, self.lstm_layer)

            return self.linear_layer(rnn_last_time_step_outputs)
        
    simple_lstm_model = EmbeddingLSTMLinearModel(embedding_matrix, 32, 2)
    simple_lstm_model.load_state_dict(torch.load('./lstm/imdb_lstm_epoch20.pt'))

    sent_val = UDA_pytorch_classifier_predict(simple_lstm_model,
                            [vocab(tokenizer_lowercase(Input_Text))],
                            rnn=True).numpy()[0]

    if sent_val == 1:
        output = 'Your review is positive :)'
    else:
        output = 'Your review is negative :('
    
    return Response(json.dumps(output))
