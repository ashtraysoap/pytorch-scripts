import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AdditiveAttention(nn.Module):

    def __init__(self, dim_k, dim_q, dropout_p=0.1):
        """Creates an additive attention layer.

        Args:
            dim_k: Dimension of the key.
            dim_q: Dimension of the query.
        """

        super(AdditiveAttention, self).__init__()
        self.linear_key = nn.Linear(dim_k, 1)
        self.linear_query = nn.Linear(dim_q, 1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        """Computes the additive attention function.

        Args:
            Q: queries of shape [batch, dim_q]
            K: keys of shape [batch, n_keys, dim_k]
            V: values of shape [batch, n_keys, dim_v]
        """
        
        n_keys = K.size()[1]
        tk = self.linear_key(K)                 # tk: [batch, n_keys, 1]
        tq = self.linear_query(Q)               # tq: [batch, 1]
        tq = tq.expand(-1, n_keys)              # tq: [batch, n_keys]
        tq = tq.unsqueeze(2)                    # tq: [batch, n_keys, 1]
        weights = self.softmax(tk + tq)
        weights = weights.permute(0, 2, 1)      # weights: [batch, 1, n_keys]
        output = torch.bmm(weights, V)          # output: [batch, 1, dim_v]
        
        weights = weights.squeeze(1)
        output = output.squeeze(1)
        return output, weights



class DotProductAttention(nn.Module):

    def __init__(self):
        pass

    def forward(self, Q, K, V):

    
        weights = F.softmax(None, dim=1)

        return

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
        o = self.out(o)
        o = self.log_softmax(o)
        
        a = attn_weights.squeeze(dim=1)
        # a : [batch, enc_dim[0]]
        # o : [1, batch, out_dim]
        # h : ?
        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder_2(nn.Module):

    def __init__(self, hid_dim, emb_dim, out_dim, key_dim, val_dim, dropout_p=0.1, **kwargs):
        super(AttentionDecoder_2, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = AdditiveAttention(key_dim, hid_dim)
        self.gru = nn.GRU(emb_dim + val_dim, hid_dim)
        self.w_h = nn.Linear(hid_dim, emb_dim)
        self.w_z = nn.Linear(val_dim, emb_dim)
        self.out = nn.Linear(emb_dim, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, annotations):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """

        emb = self.embedding(input).squeeze(dim=2)  # emb: [1, batch, emb_dim]
        emb = self.dropout(emb)

        context, attn_weights = self.attention(Q=hidden.squeeze(dim=0), 
                                    K=annotations, 
                                    V=annotations)
        
        gru_in = torch.cat((emb, context.unsqueeze(dim=0)), dim=2)  # gru_in: [1, batch, emb_dim + val_dim]
        out, hid = self.gru(gru_in, hidden)

        # deep output layer
        z = self.w_z(context)       # z: [batch, emb_dim]
        h = self.w_h(hid)           # h: [batch, emb_dim]
        out = self.out(emb.squeeze(dim=0) + z + h) # out: [batch, out_dim]

        out = self.log_softmax(out)
        return out, hid, attn_weights

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
                        n_keys=enc_seq_len)

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

        _, topi = y.topk(1, dim=2)
        topi = topi.squeeze(2)

        return {
            'token_ids': topi,
            'alignments': att,
        }