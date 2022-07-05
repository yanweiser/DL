import torch
import torch.nn as nn
import torch.nn.functional as F


class SentEncoder(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size):
        super(SentEncoder, self).__init__()
        """
        Fill this method. You can define and load the word embeddings here.
        You should define the convolution layer here, which use ReLU
        activation. Tips: You can use nn.Sequential to make your code cleaner.
        """

        self.step1 = nn.Sequential(
            nn.Embedding.from_pretrained(torch.from_numpy(pretrained_emb)),
            nn.Conv1d(50, configs['conv_dim'], 3, padding=1)
            )
        self.conv2 = nn.Conv1d(1, configs['conv_dim'], 3, padding=1)
        

    def forward(self, sent):
        """
        Fill this method. It should accept a sentence and return
        the sentence embeddings
        """

        out = self.step1(sent)
        pooled, _ = torch.max(out, dim=1)
        out2 = self.conv2(pooled.view(sent.shape[0], 1, -1))
        sent_embs, _ = torch.max(out2, dim=1)

        return sent_embs


class NLINet(nn.Module):
    def __init__(self, configs, pretrained_emb, token_size, label_size, device):
        super(NLINet, self).__init__()
        """
        Fill this method. You can define the FFNN, dropout and the sentence
        encoder here.
        """
        self.encoder = SentEncoder(configs, pretrained_emb, token_size)
        self.encoder = self.encoder.to(device)
        self.decoder = nn.Sequential(
            nn.Linear(4*pretrained_emb.shape[1], configs['mlp_hidden']),
            nn.ReLU(),
            nn.Linear(configs['mlp_hidden'], label_size)
        )

    def forward(self, premise, hypothesis):
        """
        Fill this method. It should accept a pair of sentence (premise &
        hypothesis) and return the logits.
        """

        encoding_premise = self.encoder(premise)
        encoding_hypo = self.encoder(hypothesis)
        encoding_comb = torch.cat((
            encoding_premise, 
            encoding_hypo, 
            torch.abs(encoding_premise - encoding_hypo),
            encoding_premise * encoding_hypo),
            dim = 1
            )
        ### Dropout Layer to be added ###
        out = self.decoder(encoding_comb)
        return out
