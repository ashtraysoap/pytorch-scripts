import os

import fire
import numpy as np
import torch
import torch.nn as nn

from vocab import Vocab
import models
from models import Network


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MAX_LEN = 15
HIDDEN_DIM = 512
EMB_DIM = 512
ENC_SEQ_LEN = 14 * 14
ENC_DIM = 512


def run(test_dir, 
    test_srcs, 
    checkpoint, 
    vocab, 
    out="captions.out.txt",
    batch_size=16, 
    max_seq_len=MAX_LEN,
    hidden_dim=HIDDEN_DIM,
    emb_dim=EMB_DIM,
    enc_seq_len=ENC_SEQ_LEN,
    enc_dim=ENC_DIM,
    attn_activation="relu",
    decoder=2):

    if decoder == 1:
        decoder = models.AttentionDecoder
    elif decoder == 2:
        decoder = models.AttentionDecoder_2

    # load vocabulary
    vocabulary = Vocab()
    vocabulary.load(vocab)

    # load test instances file paths
    srcs = open(test_srcs).read().strip().split('\n')
    srcs = [os.path.join(test_dir, s) for s in srcs]

    # load model
    net = Network(hid_dim=hidden_dim, out_dim=vocabulary.n_words,
        sos_token=0, eos_token=1, pad_token=2,
        emb_dim=emb_dim,
        enc_seq_len=enc_seq_len,
        enc_dim=enc_dim,
        decoder=decoder)
    net.to(DEVICE)

    net.load_state_dict(torch.load(checkpoint))
   
    net.eval()

    with torch.no_grad():

        # run inference
        num_instances = len(srcs)
        i = 0
        captions = []
        while i < num_instances:
            srcs_batch = srcs[i:i + batch_size]
            batch = _load_batch(srcs_batch)

            tokens, _ = net(batch, targets=None, max_len=max_seq_len)
            tokens = tokens.permute(1, 0, 2).detach()
            _, topi = tokens.topk(1, dim=2)
            topi = topi.squeeze(2)

            # decode token output from the model
            for j in range(len(srcs_batch)):
                c = vocabulary.tensor_to_sentence(topi[j])
                c = ' '.join(c)
                captions.append(c)

            i += len(srcs_batch)

    out_f = open(out, mode='w')
    for c in captions:
        out_f.write(c + '\n')

    return

def _load_features(fp):
    x = np.load(fp)['arr_0']

    if len(x.shape) == 3:
        dim = x.shape[2]
        x = np.reshape(x, (-1, dim))

    return torch.from_numpy(x)

def _load_batch(fps):
    x = [_load_features(fp) for fp in fps]
    return torch.stack(x, dim=0)


if __name__ == "__main__":
    fire.Fire(run)