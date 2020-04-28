import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        o = self.embedding(input).squeeze(dim=2)
        o = F.relu(o)
        o, h = self.gru(o, hidden)
        o = self.log_softmax(self.out(o))
        return o, h

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, encoder_dim=(None, None), dropout_p=0.1):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_dim = encoder_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        self.attn = nn.Linear(self.hidden_size * 2, self.encoder_dim[0])
        self.attn_combine = nn.Linear(self.hidden_size + self.encoder_dim[1], self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(hidden_size, hidden_size)
        
        self.out = nn.Linear(hidden_size, output_size)
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, encoder_output):
        """
        Shapes:
            input: [1, batch_size, 1] ([seq_len, batch, index])
            hidden: [1, batch_size, hidden_size]
            encoder_output: [batch_size, enc_dim[0], enc_dim[1]]
        """

        # print("input", input.size())
        # print("hidden0", hidden.size())
        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        # print("embedded", o.size())

        # Attention
        attn_input = torch.cat((o[0], hidden[0]), dim=1)
        # attn_input size is [batch, emb_dim + hid_dim]

        attn_energies = self.attn(attn_input)
        # attn_energies size is [batch, enc_dim[0]]
        attn_weights = F.softmax(attn_energies, dim=1)
        # attn_weights size is [batch, enc_dim[0]]

        attn_weights = attn_weights.unsqueeze(dim=1)
        # attn_weights : [batch, 1, enc_dim[0]]
        # encoder_output : [batch, enc_dim[0], enc_dim[1]]
        context = torch.bmm(attn_weights, encoder_output)
        context = context.squeeze(dim=1)
        # context : [batch, enc_dim[1]]

        o = torch.cat((o[0], context), dim=1)
        # o : [batch, emd_dim + enc_dim[1]]
        o = self.attn_combine(o)
        # o : [batch, hid_dim]
        o = o.unsqueeze(0)
        # o : [1, batch, hid_dim]

        o = F.relu(o)
        o, h = self.gru(o, hidden)
        # print("gru output ", o.size())
        # print("gru hidden ", h.size())
        o = self.out(o)
        # print("linear output projection output ", o.size())
        o = self.log_softmax(o)
        # print("softmax output ", o.size())
        
        a = attn_weights.squeeze(dim=1)
        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

class Network(nn.Module):

    def __init__(
        self,
        hidden_size,
        output_size,
        sos_token, 
        eos_token, 
        pad_token,
        teacher_forcing_rat=0.2
        ):

        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.teacher_forcing_rat = teacher_forcing_rat
        self.decoder = AttentionDecoder(hidden_size, output_size, 
            encoder_dim=(14 * 14, 512))
        self.decoder.to(device)

    def forward(self, features, targets=None, max_len=10):
        """
        Shapes:
            features: [batch_size, X, Y]
            targets: [max_len, batch_size, 1]
        """

        # print(features.size())
        # if targets is not None: print(targets.size())

        # features : [batch, enc_seq_len, enc_dim]
        batch_size = features.size()[0]
        
        y = torch.tensor([[self.sos_token]] * batch_size, device=device).view(1, batch_size, 1)
        hid = self.decoder.initHidden(batch_size=batch_size)
        
        # gradually store outputs here:
        outputs = torch.zeros(max_len, batch_size, self.output_size, device=device)

        for i in range(max_len):
            out, hid, att = self.decoder(y, hid, features)
            outputs[i] = out.squeeze(dim=0)

            _, topi = out.topk(1)

            if random.random() < self.teacher_forcing_rat \
                and targets is not None:
                y = targets[i].unsqueeze(0) # teacher force
            else:
                y = topi.detach()
        
        return outputs # output logits in shape [max_len, batch, vocab]
