import random
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from attentions import AdditiveAttention
from layers import DeepOutputLayer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionDecoder_1(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        key_dim, 
        val_dim, 
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_1, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attention = attention(dim_k=key_dim, dim_q=hid_dim, hid_dim=hid_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.gru = nn.GRU(emb_dim + key_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
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
        emb = o

        context, a = self.attention(Q=hidden.squeeze(0), 
                    K=annotations, 
                    V=annotations)

        o = torch.cat((o[0], context), dim=1)
        o = o.unsqueeze(0)

        o, h = self.gru(o, hidden)

        if self.deep_out:
            o = self.out(y=emb, h=o, z=context)
        else:
            o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)


class AttentionDecoder_2(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        key_dim, 
        val_dim, 
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_2, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attention = attention(dim_k=key_dim, dim_q=hid_dim, hid_dim=hid_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.attn_combine = nn.Linear(emb_dim + val_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
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
        emb = o

        context, a = self.attention(Q=hidden.squeeze(0), 
                    K=annotations, 
                    V=annotations)

        o = torch.cat((o[0], context), dim=1)

        o = self.attn_combine(o)

        o = o.unsqueeze(0)
        o = F.relu(o)

        o, h = self.gru(o, hidden)

        if self.deep_out:
            o = self.out(y=emb, h=o, z=context)
        else:
            o = self.out(o)
        o = self.log_softmax(o)

        return o, h, a

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)

class AttentionDecoder_3(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        key_dim, 
        val_dim, 
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_3, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attention = attention(dim_k=key_dim, dim_q=hid_dim, hid_dim=hid_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.attn_combine = nn.Linear(emb_dim + val_dim, hid_dim)
        self.gru = nn.GRU(emb_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
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
        emb = o

        _, h_t = self.gru(o, hidden)

        c_t, a_t = self.attention(Q=h_t.squeeze(0), 
                    K=annotations, 
                    V=annotations)

        o_t = torch.cat((h_t.squeeze(0), c_t), dim=1)

        o_t = self.attn_combine(o_t)
        o_t = o_t.unsqueeze(0)
        o_t = F.relu(o_t)


        if self.deep_out:
            out = self.out(y=emb, h=o_t, z=c_t)
        else:
            out = self.out(o_t)
        logits = self.log_softmax(out)

        return logits, h_t, a_t

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hid_dim, device=device)


class AttentionDecoder_4(nn.Module):

    def __init__(self, 
        hid_dim, 
        emb_dim, 
        out_dim, 
        key_dim, 
        val_dim, 
        attn_activation, 
        dropout_p=0.1, 
        attention=AdditiveAttention, 
        deep_out=False, 
        **kwargs):

        super(AttentionDecoder_4, self).__init__()
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(out_dim, emb_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        
        self.attention = attention(dim_k=key_dim, dim_q=hid_dim, hid_dim=hid_dim, 
                dropout_p=dropout_p, activation=attn_activation)
        
        self.attn_combine = nn.Linear(hid_dim + val_dim, hid_dim)
        self.gru = nn.GRU(hid_dim + emb_dim, hid_dim)

        if deep_out:
            self.out = DeepOutputLayer(out_dim, emb_dim, hid_dim, val_dim)
            self.deep_out = True
        else:
            self.out = nn.Linear(hid_dim, out_dim)
            self.deep_out = False
        
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden, annotations):
        """
        Args:
            input: [1, batch, 1]
            hidden: [1, batch, hid_dim]
            annotations: [batch, n_keys, key_dim]
        """
        prev_h, prev_o = hidden

        o = self.embedding(input).squeeze(dim=2)
        o = self.dropout(o)
        emb = o

        in_t = torch.cat((o, prev_o), dim=2)

        _, h_t = self.gru(in_t, prev_h)

        c_t, a_t = self.attention(Q=h_t.squeeze(0), 
                    K=annotations, 
                    V=annotations)

        o_t = torch.cat((h_t.squeeze(0), c_t), dim=1)

        o_t = self.attn_combine(o_t)
        o_t = o_t.unsqueeze(0)
        o_t = F.relu(o_t)

        if self.deep_out:
            out = self.out(y=emb, h=o_t, z=c_t)
        else:
            out = self.out(o_t)
        logits = self.log_softmax(out)

        return logits, (h_t, o_t), a_t

    def initHidden(self, batch_size):
        h_0 = torch.zeros(1, batch_size, self.hid_dim, device=device)
        o_0 = torch.zeros(1, batch_size, self.hid_dim, device=device)
        return (h_0, o_0)



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
        deep_out=False,
        attn_activation="relu",
        decoder=AttentionDecoder_1,
        attention=AdditiveAttention
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
        
        print(decoder)
        print(attention)
        
        self.decoder = decoder(hid_dim=hid_dim, 
                        emb_dim=emb_dim, 
                        out_dim=out_dim, 
                        key_dim=enc_dim, 
                        val_dim=enc_dim,
                        n_keys=enc_seq_len,
                        attn_activation=attn_activation,
                        deep_out=deep_out,
                        dropout_p=dropout_p,
                        attention=attention)

        self.decoder.to(device)

    def forward(self, features, targets=None, max_len=10, teacher_forcing_rat=None):
        """
        Shapes:
            features: [batch_size, X, Y]
            targets: [max_len, batch_size, 1]
        """

        if teacher_forcing_rat == None:
            teacher_forcing_rat = self.teacher_forcing_rat

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

            if random.random() < teacher_forcing_rat \
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