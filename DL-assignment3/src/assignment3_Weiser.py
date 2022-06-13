# -*- coding: utf-8 -*-
"""
    Deep Learning for NLP
    Assignment 3: Language Identification using Recurrent Architectures
    Based on original code by Hande Celikkanat & Miikka Silfverberg
"""


from random import choice, random, shuffle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk

from data import read_datasets, WORD_BOUNDARY, UNK, HISTORY_SIZE
from paths import data_dir

torch.set_num_threads(10)

if torch.cuda.is_available():
    my_device = torch.device('cuda')
    cuda = True
else:
    my_device = torch.device('cpu')
    cuda = False


#--- hyperparameters ---
N_EPOCHS = 100
LEARNING_RATE = 3e-4
REPORT_EVERY = 5
EMBEDDING_DIM = 30
HIDDEN_DIM = 64
BATCH_SIZE = 100
N_LAYERS = 2


#--- models ---
class LSTMModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.embed = nn.Embedding(self.character_set_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.decoder = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        # Recommendation: use a single input for lstm layer (no special initialization of the hidden layer):
        lstm_out, hidden = self.lstm(embeds)
        logits = self.decoder(lstm_out[-1].view(len(inputs[0]),self.hidden_dim))
        return F.log_softmax(logits, dim=1)


class GRUModel(nn.Module): 
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(GRUModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.embed = nn.Embedding(self.character_set_size, self.embedding_dim)
        self.gru = nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.decoder = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        # Recommendation: use a single input for gru layer (no special initialization of the hidden layer):
        gru_out, hidden = self.gru(embeds)
        logits = self.decoder(gru_out[-1].view(len(inputs[0]),self.hidden_dim))
        return F.log_softmax(logits, dim=1)
        


class RNNModel(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 character_set_size,
                 n_layers,
                 hidden_dim,
                 n_classes):
        super(RNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_set_size = character_set_size        
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.embed = nn.Embedding(self.character_set_size, self.embedding_dim)
        self.rnn = nn.RNN(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers)
        self.decoder = nn.Linear(in_features=self.hidden_dim, out_features=self.n_classes)

    def forward(self, inputs):
        embeds = self.embed(inputs)
        # Recommendation: use a single input for rnn layer (no special initialization of the hidden layer):
        rnn_out, hidden = self.rnn(embeds)
        logits = self.decoder(rnn_out[-1].view((inputs[0]),self.hidden_dim))
        return F.log_softmax(logits, dim=1)



# --- auxilary functions ---
def get_minibatch(minibatchwords, character_map, languages):
    seq_lens = [get_word_length(word) for word in minibatchwords]
    sl = max(seq_lens) + 2
    size = len(minibatchwords) # last batch might be smaller than BATCH_SIZE

    mb_x = torch.empty((sl,size), dtype=torch.long, device=my_device)
    mb_y = torch.empty(size, dtype=torch.long, device=my_device)

    for i in range(size):
        word_len = len(minibatchwords[i]['TENSOR'])
        mb_x[:,i] = F.pad(minibatchwords[i]['TENSOR'], (0,sl-word_len), 'constant', character_map['#'])
        mb_y[i] = label_to_idx(minibatchwords[i]['LANGUAGE'], languages)

    return mb_x, mb_y


def label_to_idx(lan, languages):
    languages_ordered = list(languages)
    languages_ordered.sort()
    return torch.LongTensor([languages_ordered.index(lan)]).to(my_device)


def get_word_length(word_ex):
    return len(word_ex['WORD'])


def evaluate(dataset,model,eval_batch_size,character_map,languages):
    correct = 0
    
    # WRITE CODE HERE IF YOU LIKE
    for i in range(0,len(dataset),eval_batch_size):
        minibatchwords = dataset[i:i+eval_batch_size]    
        mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)

        with torch.no_grad():
            preds = model(mb_x)
            correct += torch.eq(preds.argmax(dim=1), mb_y).sum(dim=0).item()
        # WRITE CODE HERE

    return correct * 100.0 / len(dataset)



if __name__=='__main__':

    # --- select the recurrent layer according to user input ---
    if len(sys.argv) < 2:
        print('-------')
        print('You didn''t provide any arguments!')
        print('Using LSTM model as default')
        print('To select a model, call the program with one of the arguments: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('-------')
        model_choice = 'lstm'
    elif len(sys.argv) == 2:
        print('-------')
        print('Running with ' + sys.argv[1][1:] + ' model')
        print('-------')        
        model_choice = sys.argv[1][1:]
    else:
        print('-------')
        print('Wrong number of arguments')
        print('Please call the model with exactly one argument, which can be: -lstm, -gru, -rnn')
        print('Example: python assignment3_LOAICIGA.py -gru')
        print('Using LSTM model as default')
        print('-------')        
        model_choice = 'lstm'


    #--- initialization ---

    if BATCH_SIZE == 1:
        data, character_map, languages = read_datasets('uralic.mini',data_dir)
    else:
        data, character_map, languages = read_datasets('uralic',data_dir)

    trainset = [datapoint for lan in languages for datapoint in data['training'][lan]]
    n_languages = len(languages)
    character_set_size = len(character_map)

    model = None
    if model_choice == 'lstm':
        model = LSTMModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'gru':
        model = GRUModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)
    elif model_choice == 'rnn':
        model = RNNModel(EMBEDDING_DIM,
                                    character_set_size,
                                    N_LAYERS,
                                    HIDDEN_DIM,
                                    n_languages)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.NLLLoss()
    model.to(my_device)
    model.train()

    # --- training loop ---
    for epoch in range(N_EPOCHS):
        total_loss = 0

        # Generally speaking, it's a good idea to shuffle your
        # datasets once every epoch.
        shuffle(trainset)

        trainset = sorted(trainset, key=get_word_length)
        # Sort your training set according to word-length, 
        # so that similar-length words end up near each other
        # You can use the function get_word_length as your sort key.
               
        for i in range(0,len(trainset),BATCH_SIZE):
            minibatchwords = trainset[i:i+BATCH_SIZE]

            # print(minibatchwords)

            mb_x, mb_y = get_minibatch(minibatchwords, character_map, languages)

            output = model(mb_x)
            optimizer.zero_grad()
            loss = loss_function(output, mb_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print('epoch: %d, loss: %.4f' % ((epoch+1), total_loss))

        if ((epoch+1) % REPORT_EVERY) == 0:
            model.eval()
            train_acc = evaluate(trainset,model,BATCH_SIZE,character_map,languages)
            dev_acc = evaluate(data['dev'],model,BATCH_SIZE,character_map,languages)
            print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' % 
                  (epoch+1, total_loss, train_acc, dev_acc))
            model.train()

        
    # --- test ---    
    model.eval()

    test_acc = evaluate(data['test'],model,BATCH_SIZE,character_map,languages)        
    print('test acc: %.2f%%' % (test_acc))
