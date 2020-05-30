import math

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdditiveAttention(nn.Module):

    def __init__(self, dim_k, dim_q, hid_dim, dropout_p=0.1, activation="relu", **kwargs):
        """Creates an additive attention layer.

        Args:
            dim_k: Dimension of the key.
            dim_q: Dimension of the query.
            hid_dim: Dimension of the hidden layer.
        """

        super(AdditiveAttention, self).__init__()
        self.linear_key = nn.Linear(dim_k, hid_dim)
        self.linear_query = nn.Linear(dim_q, hid_dim)
        self.hidden = nn.Linear(hid_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()

        self.dropout_p = dropout_p

    def forward(self, Q, K, V):
        """Computes the additive attention function.

        Args:
            Q: queries of shape [batch, dim_q]
            K: keys of shape [batch, n_keys, dim_k]
            V: values of shape [batch, n_keys, dim_v]
        """
        
        n_keys = K.size()[1]
        tk = self.linear_key(K)                 # tk: [batch, n_keys, hid_dim]
        tk = F.dropout(tk, p=self.dropout_p)

        tq = self.linear_query(Q)               # tq: [batch, hid_dim]
        tq = F.dropout(tq, p=self.dropout_p)
        tq = tq.unsqueeze(1)                    # tq: [batch, 1, hid_dim]
        tq = tq.expand(-1, n_keys, -1)          # tq: [batch, n_keys, hid_dim]
        
        tqk = self.activation(tk + tq)
        
        energies = self.hidden(tqk)         # energies: [batch, n_keys, 1]

        weights = self.softmax(energies)        # weights: [batch, n_keys, 1]
        weights = weights.permute(0, 2, 1)      # weights: [batch, 1, n_keys]
        output = torch.bmm(weights, V)          # output: [batch, 1, dim_v]
                
        weights = weights.squeeze(1)
        output = output.squeeze(1)
        return output, weights



class GeneralAttention(nn.Module):

    def __init__(self, dim_q, dim_k, **kwargs):
        super(GeneralAttention, self).__init__()

        self.linear_k = nn.Linear(dim_k, dim_q)

    def forward(self, Q, K, V):
        """Computes the 'general' attention function.

        Args:
            Q: queries of shape [batch, dim_q]
            K: keys of shape [batch, n_keys, dim_k]
            V: values of shape [batch, n_keys, dim_v]
        """
        
        keys = self.linear_k(K)             # keys : [batch, n_keys, dim_q]
        keys = keys.permute(0, 2, 1)
        query = Q.unsqueeze(1)              # query: [batch, 1, dim_q]
        energies = torch.bmm(query, keys)   # energies : [batch, 1, n_keys]
        weights = F.softmax(energies, dim=2)
        output = torch.bmm(weights, V)

        weights = weights.squeeze(1)
        output = output.squeeze(1)
        return output, weights


class ScaledGeneralAttention(nn.Module):

    def __init__(self, dim_q, dim_k, **kwargs):
        super(ScaledGeneralAttention, self).__init__()

        self.linear_k = nn.Linear(dim_k, dim_q)
        self.scale = 1 / math.pow(dim_k, 0.5)

    def forward(self, Q, K, V):
        """Computes the scaled 'general' attention function.

        Args:
            Q: queries of shape [batch, dim_q]
            K: keys of shape [batch, n_keys, dim_k]
            V: values of shape [batch, n_keys, dim_v]
        """
        
        keys = self.linear_k(K)             # keys : [batch, n_keys, dim_q]
        keys = keys.permute(0, 2, 1)
        query = Q.unsqueeze(1)              # query: [batch, 1, dim_q]
        energies = torch.bmm(query, keys)   # energies : [batch, 1, n_keys]
        energies = energies * self.scale
        weights = F.softmax(energies, dim=2)
        output = torch.bmm(weights, V)

        weights = weights.squeeze(1)
        output = output.squeeze(1)
        return output, weights
