# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, hidden_size, embedding_matrix, num_classes, num_recur_layer=1):
        super(NeuralNetwork, self).__init__()
        self.embedding_layer, emb_dim = self.init_embedding_layer(embedding_matrix)
        self.hidden_size = hidden_size
        self.num_recur_layer = num_recur_layer
        self.gru = nn.GRU(emb_dim, hidden_size, num_recur_layer, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # try using softmax

    def init_embedding_layer(self, embedding_matrix, non_trainable=False):
        num_embeddings, embedding_dim = embedding_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': embedding_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = False
        return emb_layer, embedding_dim

    def forward(self, input, hidden):
        embed_info = self.embedding_layer(input)
        output, last_state = self.gru(embed_info, hidden)
        # Output is defined using shape (batch_size, state, hidden_dimension), thus we need to take the last state (-1)
        return self.fc(output[:, -1, :])

    def init_hidden(self, batch_size):
        return torch.Variable(torch.zeros(self.num_recur_layer, batch_size, self.hidden_size))
