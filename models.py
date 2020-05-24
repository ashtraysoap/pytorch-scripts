import random
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from attentions import AdditiveAttention
from layers import DeepOutputLayer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):

    def __init__(self, hid_dim, out_dim, dropout_p=0.1):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.out_dim, self.hid_dim)
        
        self.gru = nn.GRU(hid_dim, hid_dim)
        
        self.out = nn.Linear(hid_dim, out_dim)
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        o = self.embedding(input).squeeze(dim=2)
        o = F.relu(o)
        o, h = self.gru(o, hidden)
        o = self.log_softmax(self.out(o))
        return o, h

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder(nn.Module):

    def __init__(self, hid_dim, out_dim, n_keys, key_dim, dropout_p=0.1, **kwargs):
        super(AttentionDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_keys = n_keys
        self.key_dim = key_dim
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.out_dim, self.hid_dim)
        
        self.attn = nn.Linear(self.hid_dim * 2, n_keys)
        self.attn_combine = nn.Linear(self.hid_dim + key_dim, self.hid_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.gru = nn.GRU(hid_dim, hid_dim)
        #self.gru = nn.GRU(hid_dim + key_dim, hid_dim)
        
        self.out = nn.Linear(hid_dim, out_dim)
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, encoder_output):
        """
        Shapes:
            input: [1, batch_size, 1] ([seq_len, batch, index])
            hidden: [1, batch_size, hid_dim]
            encoder_output: [batch_size, enc_dim[0], enc_dim[1]]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)

        attn_input = torch.cat((o[0], hidden[0]), dim=1)

        attn_energies = self.attn(attn_input)
        attn_weights = F.softmax(attn_energies, dim=1)

        attn_weights = attn_weights.unsqueeze(dim=1)
        context = torch.bmm(attn_weights, encoder_output)
        context = context.squeeze(dim=1)
        a = attn_weights.squeeze(dim=1)
        # context : [batch, enc_dim[1]]

        o = torch.cat((o[0], context), dim=1)
        # o : [batch, emd_dim + enc_dim[1]]
        
        o = self.attn_combine(o)
        o = o.unsqueeze(0)

        o = F.relu(o)
        o, h = self.gru(o, hidden)

        o = self.out(o)
        o = self.log_softmax(o)
        
        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder_2(nn.Module):

    def __init__(self, hid_dim, emb_dim, out_dim, key_dim, val_dim, attn_activation, dropout_p=0.1, **kwargs):
        super(AttentionDecoder_2, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = AdditiveAttention(key_dim, hid_dim, hid_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        self.attn_combine = nn.Linear(emb_dim + val_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim)

        #self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
        self.out = nn.Linear(hid_dim, out_dim)
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, annotations):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)

        #q = torch.cat((hidden, o), dim=2).squeeze(0)
        # context, a = self.attention(Q=hidden.squeeze(dim=0), 
        #                             K=annotations, 
        #                             V=annotations)
        context, a = self.attention(Q=hidden.squeeze(0), 
                    K=annotations, 
                    V=annotations)

        o = torch.cat((o[0], context), dim=1)
        #o = F.dropout(o, p=self.dropout_p)

        o = self.attn_combine(o)
        #o = F.dropout(o, p=self.dropout_p)

        o = o.unsqueeze(0)
        o = F.relu(o)

        o, h = self.gru(o, hidden)
        #o = F.dropout(o, p=self.dropout_p)

        #out = self.out(y=emb, h=hid, z=context)
        o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)


class Network(nn.Module):

    def __init__(
        self,
        hid_dim,
        out_dim,
        emb_dim,
        enc_seq_len,
        enc_dim,
        sos_token, 
        eos_token, 
        pad_token,
        teacher_forcing_rat=0.2,
        dropout_p=0.1,
        attn_activation="relu",
        decoder=AttentionDecoder
        ):

        super(Network, self).__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.emb_dim = emb_dim
        self.enc_seq_len = enc_seq_len
        self.enc_dim = enc_dim
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.teacher_forcing_rat = teacher_forcing_rat

        self.decoder = decoder(hid_dim=hid_dim, 
                        emb_dim=emb_dim, 
                        out_dim=out_dim, 
                        key_dim=enc_dim, 
                        val_dim=enc_dim,
                        n_keys=enc_seq_len,
                        attn_activation=attn_activation,
                        dropout_p=dropout_p)

        self.decoder.to(device)

    def forward(self, features, targets=None, max_len=10):
        """
        Shapes:
            features: [batch_size, X, Y]
            targets: [max_len, batch_size, 1]
        """

        # features : [batch, enc_seq_len, enc_dim]
        batch_size = features.size()[0]
        
        y = torch.tensor([[self.sos_token]] * batch_size, device=device).view(1, batch_size, 1)
        hid = self.decoder.initHidden(batch_size=batch_size)
        
        # gradually store outputs here:
        outputs = torch.zeros(max_len, batch_size, self.out_dim, device=device)
        attentions = torch.zeros(max_len, batch_size, self.enc_seq_len, device=device)

        for i in range(max_len):
            out, hid, att = self.decoder(y, hid, features)
            outputs[i] = out.squeeze(dim=0)
            attentions[i] = att

            _, topi = out.topk(1)

            if random.random() < self.teacher_forcing_rat \
                and targets is not None:
                y = targets[i].unsqueeze(0) # teacher force
            else:
                y = topi.detach()
        
        return outputs, attentions # output logits in shape [max_len, batch, vocab]



    def infere(self, features, max_len=10):
        out, att = self.forward(features, targets=None, max_len=max_len)
        out = out.permute(1, 0, 2).detach()
        att = att.permute(1, 0, 2).detach()

        _, topi = out.topk(1, dim=2)
        topi = topi.squeeze(2)

        return {
            'token_ids': topi,
            'alignments': att,
        }