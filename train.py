# -*- coding: utf-8 -*-
import pandas as pd
import torch

from modules.neural_network import NeuralNetwork
from modules.word_embedding import WordEmbedding
from modules.configuration import DATA_TRAIN_FILE, WORD2VEC_FILE_SLIM
from modules.pre_process import PreProcessStage

# Hyper-parameters
sequence_length = 28
input_size = 300
hidden_size = 50
num_layers = 2
# 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
num_classes = 6
batch_size = 25
num_epochs = 2
learning_rate = 0.003

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset_df = pd.read_csv(DATA_TRAIN_FILE)
pre_process = PreProcessStage(dataset_df)
pre_process.pre_process_dataset()

embedding_model = WordEmbedding(WORD2VEC_FILE_SLIM)
embedding_matrix = embedding_model.load_from_vocabulary(pre_process.word_2_idx)

model = NeuralNetwork(hidden_size, embedding_matrix, num_classes)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for input, label in zip(pre_process.input_data, pre_process.labels):
        print(input)
        print(label)
