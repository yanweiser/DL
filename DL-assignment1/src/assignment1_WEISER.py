# -*- coding: utf-8 -*-
"""
   Deep Learning for NLP
   Assignment 1: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Remember to use PyTorch for your NN implementation.
   Original code by Hande Celikkanat & Miikka Silfverberg. Minor modifications by Sharid Loáiciga.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os
import random

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir


# name of the embeddings file to use
# Alternatively, you can also use the text file GoogleNews-pruned2tweets.txt (from Moodle),
# or the full set, wiz. GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/) 
embeddings_file = 'GoogleNews-pruned2tweets.bin'


#--- hyperparameters ---

# Feel free to experiment with different hyperparameters to see how they compare! 
# You can turn in your assignment with the best settings you find.

n_classes = len(LABEL_INDICES)
n_epochs = 40
learning_rate = 3e-4
# report_every = 1
verbose = False
hp_tune = False

random.seed(1)
torch.manual_seed(0)

my_device = torch.device('cpu')

# since the computation is not vectorized, its faster to run it on the cpu

# if torch.cuda.is_available():
#       my_device = torch.device('cuda')
#       cuda = True
# else:
#       my_device = torch.device('cpu')
#       cuda = False



#--- auxilary functions ---

# To convert string label to pytorch format:
def label_to_idx(label):
  return torch.LongTensor([LABEL_INDICES[label]]).to(my_device)

# return a FloatTensor that is the sum of the word embeddings of the input tweet
# additional inputs are the token to index (tti) dictionary and the index to embedding (ite) dictionary
def tweet_to_embed(tweet, tti, ite):
      temp = np.zeros((len(tweet), 300), dtype='float')
      for i, token in enumerate(tweet):
        if token in tti:
          temp[i] = ite[tti.get(token)]
      tensor = torch.FloatTensor(temp)
      return tensor.sum(dim=0).to(my_device)

def train(model, data, optimizer, loss_function, report_every = 10):
      
  # convert tweets to embeddings once instead of every epoch
  embeds = []
  for tweet in data['training']:
        embeds.append({'BODY': tweet_to_embed(tweet['BODY'], word_to_idx, pretrained_embeds), 'SENTIMENT': label_to_idx(tweet['SENTIMENT'])})
  k = len(embeds)
  for epoch in range(n_epochs):
    training_losses = []
    total_loss = 0
    epoch_order = random.sample(embeds, k) # shuffle the order of tweets every epoch to better generalize
    for tweet in epoch_order: 
      preds = model(tweet['BODY'])
      optimizer.zero_grad()
      loss = loss_function(torch.reshape(preds, (1,3)), tweet['SENTIMENT'])
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
    if ((epoch+1) % report_every) == 0:
      print('epoch: %d,\tloss: %.4f' % (epoch+1, total_loss*100/len(data['training'])))

def test(model, data, kind):
  correct = 0
  with torch.no_grad():
    for tweet in data[kind]:
      gold_class = label_to_idx(tweet['SENTIMENT'])
      sent = tweet_to_embed(tweet['BODY'], word_to_idx, pretrained_embeds)

      preds = model(sent)
      predicted = torch.argmax(preds, dim=0)
      correct += torch.eq(predicted,gold_class).item()
  return correct/len(data[kind])

#--- model ---

class FFNN(nn.Module):
  # Feel free to add whichever arguments you like here.
  # Note that pretrained_embeds is a numpy matrix of shape (num_embeddings, embedding_dim)
  def __init__(self, pretrained_embeds, n_classes, hid_size1=64, hid_size2=64):
      super(FFNN, self).__init__()
      self.l1 = nn.Linear(pretrained_embeds.shape[1], hid_size1, device=my_device)
      self.l2 = nn.Linear(hid_size1, hid_size2, device=my_device)
      self.l3 = nn.Linear(hid_size2, n_classes, device=my_device)
      self.relu_act = nn.ReLU()
      self.log_sm = nn.LogSoftmax(dim=0)
      

  def forward(self, x):
      h1 = self.relu_act(self.l1(x))
      h2 = self.relu_act(self.l2(h1))
      logits = self.log_sm(self.l3(h2))
      return logits


#--- "main" ---

if __name__=='__main__':
  #--- data loading ---
  data = read_semeval_datasets(data_dir)
  gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file), binary=True)
  pretrained_embeds = gensim_embeds.vectors
  # To convert words in the input tweet to indices of the embeddings matrix:
  word_to_idx = {word: i for i, word in enumerate(gensim_embeds.vocab.keys())}

  #--- set up ---
  # WRITE CODE HERE
  model = FFNN(pretrained_embeds, n_classes, hid_size1=204, hid_size2=288)
  loss_function = nn.NLLLoss()
  optimizer = optim.SGD(model.parameters(), lr=6e-4, momentum=0.86)
  

  #--- training ---
  
  train(model, data, optimizer, loss_function, report_every=1)


  #--- testing ---

  train_acc = test(model, data, 'training')
  val_acc = test(model, data, 'development.gold')
  test_acc = test(model, data, 'test.gold')
  print('Accuracies:\n train = {:.2f}\n val = {:.2f}\n test = {:.2f}'.format(100*train_acc, 100*val_acc, 100*test_acc))
    
  #--- optimizing ---
  # Feel free to use the development data to tune hyperparameters if you like!
  # Parameters to tune : learning rate, hidden size 1, hidden size 2
  # random search, ranges:
  if hp_tune:
    lr_range = [5e-4, 5e-3]
    h1_range = [200, 300]
    h2_range = [200, 300]
    mom_range = [0.85, 0.90]

    NUM_TRIALS = 20
    trial_vars = {}
    best = 0
    for i in range(NUM_TRIALS):
      if i > 0:
            print('------------------------------')
      trial_vars['h1_trial'] = random.randint(h1_range[0], h1_range[1])
      trial_vars['h2_trial'] = random.randint(h2_range[0], h2_range[1])
      trial_vars['lr_trial'] = random.uniform(lr_range[0], lr_range[1])
      # trial_vars['lr_trial'] = 0.0006314250778068356
      trial_vars['mom_trial'] = random.uniform(mom_range[0], mom_range[1])
      print(f'Trial {i+1}:\n Learning rate = {trial_vars["lr_trial"]}\n hidden size 1 = {trial_vars["h1_trial"]}\n hidden size 2 = {trial_vars["h2_trial"]}\n momentum = {trial_vars["mom_trial"]}')
      trial_model = FFNN(pretrained_embeds, n_classes, hid_size1=trial_vars['h1_trial'], hid_size2=trial_vars['h2_trial'])
      trial_loss_function = nn.NLLLoss()
      trial_optimizer = optim.SGD(trial_model.parameters(), lr=trial_vars['lr_trial'], momentum=trial_vars['mom_trial'])

      train(trial_model, data, trial_optimizer, trial_loss_function, report_every=100)

      train_acc = test(trial_model, data, 'training')
      val_acc = test(trial_model, data, 'development.gold')
      test_acc = test(trial_model, data, 'test.gold')
      print('Accuracies:\n train = {:.2f}\n val = {:.2f}\n test = {:.2f}'.format(100*train_acc, 100*val_acc, 100*test_acc))
      performance = 0.8*train_acc+2*val_acc # arbitrary measurement of how well the model performs mainly based on the validation accuracy
      print('Performance = {}'.format(performance))
      if performance > best:
            best = performance
            best_vars = trial_vars
    print('\n\n------------------------------')
    print('Best Model parameters:')
    print(f' Learning rate = {best_vars["lr_trial"]}\n hidden size 1 = {best_vars["h1_trial"]}\n hidden size 2 = {best_vars["h2_trial"]}\n momentum = {best_vars["mom_trial"]}')

