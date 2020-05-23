import os
import sys
import time

import fire
import torch
import torch.nn as nn

from data_loader import DataLoader
from prepro import normalize_strings, filter_inputs
from vocab import Vocab

import models
from models import Network


MAX_LEN = 15
HIDDEN_DIM = 512
EMB_DIM = 512
ENC_SEQ_LEN = 14 * 14
ENC_DIM = 512
EPOCHS = 100
BATCH_SIZE = 4
CLIP_VAL = 1
TEACHER_FORCE_RAT = 0.2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:\t", DEVICE)

def run(train_feats, 
    train_caps, 
    val_feats=None, 
    val_caps=None, 
    train_prefix="",
    val_prefix="",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    max_seq_len=MAX_LEN,
    hidden_dim=HIDDEN_DIM,
    emb_dim=EMB_DIM,
    enc_seq_len=ENC_SEQ_LEN,
    enc_dim=ENC_DIM,
    clip_val=CLIP_VAL,
    teacher_force=TEACHER_FORCE_RAT,
    dropout_p=0.1,
    attn_activation="relu",
    epsilon=0.0005,
    checkpoint="",
    out_dir="Pytorch_Exp_Out",
    decoder=1):

    if decoder == 1:
        decoder = models.AttentionDecoder
    elif decoder == 2:
        decoder = models.AttentionDecoder_2

    train(train_feats, train_caps, val_feats, val_caps, train_prefix, 
        val_prefix, epochs, batch_size, max_seq_len, hidden_dim, emb_dim,
        enc_seq_len, enc_dim, clip_val,
        teacher_force, dropout_p, attn_activation, epsilon, 
        checkpoint, out_dir, decoder)


def train(train_feats, 
    train_caps, 
    val_feats, 
    val_caps, 
    train_prefix="",
    val_prefix="",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    max_seq_len=MAX_LEN,
    hidden_dim=HIDDEN_DIM,
    emb_dim=EMB_DIM,
    enc_seq_len=ENC_SEQ_LEN,
    enc_dim=ENC_DIM,
    clip_val=CLIP_VAL,
    teacher_force=TEACHER_FORCE_RAT,
    dropout_p=0.1,
    attn_activation="relu",
    epsilon=0.0005,
    checkpoint="",
    out_dir="Pytorch_Exp_Out",
    decoder=None):
    
    print("EXPERIMENT START ", time.asctime())

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 1. Load the data

    train_captions = open(train_caps, mode='r', encoding='utf-8') \
        .read().strip().split('\n')
    train_features = open(train_feats, mode='r').read().strip().split('\n')
    train_features = [os.path.join(train_prefix, z) for z in train_features]    
    
    assert len(train_captions) == len(train_features)

    if val_caps:
        val_captions = open(val_caps, mode='r', encoding='utf-8') \
            .read().strip().split('\n')

        val_features = open(val_feats, mode='r').read().strip().split('\n')
        val_features = [os.path.join(val_prefix, z) for z in val_features]

        assert len(val_captions) == len(val_features)
    
    # 2. Preprocess the data

    train_captions = normalize_strings(train_captions)
    train_data = list(zip(train_captions, train_features))
    train_data = filter_inputs(train_data)
    print("Total training instances: ", len(train_data))

    if val_caps:
        val_captions = normalize_strings(val_captions)
        val_data = list(zip(val_captions, val_features))
        val_data = filter_inputs(val_data)
        print("Total validation instances: ", len(val_data))
    
    vocab = Vocab()
    vocab.build_vocab(map(lambda x: x[0], train_data), max_size=10000)
    vocab.save(path=os.path.join(out_dir, 'vocab.txt'))
    print("Vocabulary size: ", vocab.n_words)

    # 3. Initialize the network, optimizer & loss function

    net = Network(hid_dim=hidden_dim, out_dim=vocab.n_words,
        sos_token=0, eos_token=1, pad_token=2,
        teacher_forcing_rat=teacher_force,
        emb_dim=emb_dim,
        enc_seq_len=enc_seq_len,
        enc_dim=enc_dim,
        dropout_p=dropout_p,
        decoder=decoder)
    net.to(DEVICE)

    if checkpoint:
        net.load_state_dict(torch.load(checkpoint))
    
    optimizer = torch.optim.Adam(net.parameters())
    loss_function = nn.NLLLoss()

    # 4. Train

    prev_val_l = sys.maxsize
    total_instances = 0
    total_steps = 0
    train_loss_log = []
    train_loss_log_batches = []
    val_loss_log = []
    val_loss_log_batches = []

    train_data = DataLoader(captions=map(lambda x: x[0], train_data),
        sources=map(lambda x: x[1], train_data), batch_size=batch_size, 
        vocab=vocab, max_seq_len=max_seq_len)

    if val_caps:
        val_data = DataLoader(captions=map(lambda x: x[0], val_data),
            sources=map(lambda x: x[1], val_data), batch_size=batch_size, 
            vocab=vocab, max_seq_len=max_seq_len)

    training_start_time = time.time()

    for e in range(1, epochs + 1):
        print("Epoch ", e)

        # train one epoch
        train_l, inst, steps, t, l_log = train_epoch(model=net, loss_function=loss_function,
            optimizer=optimizer, data_iter=train_data, max_len=max_seq_len, clip_val=clip_val,
            epsilon=epsilon)
        
        # epoch logs
        print("Training loss:\t", train_l)
        print("Instances:\t", inst)
        print("Steps:\t", steps)
        hours = t // 3600
        mins = (t % 3600) // 60
        secs = (t % 60)
        print("Time:\t{0}:{1}:{2}".format(hours, mins, secs))
        total_instances += inst
        total_steps += steps
        train_loss_log.append(train_l)
        train_loss_log_batches += l_log
        print()


        # evaluate
        if val_caps:
            val_l, l_log = evaluate(model=net, loss_function=loss_function, 
                data_iter=val_data, max_len=max_seq_len, epsilon=epsilon)

            # validation logs
            print("Validation loss: ", val_l)
            if val_l < prev_val_l:
                torch.save(net.state_dict(), os.path.join(out_dir, 'net.pt'))
            val_loss_log.append(val_l)
            val_loss_log_batches += l_log


        #sample model
        samples = sample(net, train_data, vocab, samples=3, max_len=max_seq_len)
        for t, s in samples:
            print("Target:\t", t)
            print("Predicted:\t", s)
            print()

    # Experiment summary logs.
    tot_time = time.time() - training_start_time
    hours = tot_time // 3600
    mins = (tot_time % 3600) // 60
    secs = (tot_time % 60)
    print("Total training time:\t{0}:{1}:{2}".format(hours, mins, secs))
    print("Total training instances:\t", total_instances)
    print("Total training steps:\t", total_steps)
    print()

    _write_loss_log("train_loss_log.txt", out_dir, train_loss_log)
    _write_loss_log("train_loss_log_batches.txt", out_dir, train_loss_log_batches)

    if val_caps:
        _write_loss_log("val_loss_log.txt", out_dir, val_loss_log)
        _write_loss_log("val_loss_log_batches.txt", out_dir, val_loss_log_batches)

    print("EXPERIMENT END ", time.asctime())

def train_epoch(model, loss_function, optimizer, data_iter, max_len=MAX_LEN, 
    clip_val=CLIP_VAL, epsilon=0.0005):
    """Trains the model for one epoch.

    Returns:
        The epoch loss, number of instances processed, number of optimizer 
        steps performed, duration of the epoch, list of losses for each batch.
    """

    # set the network to training mode
    model.train()

    total_loss = 0
    loss_log = []
    num_instances = 0
    num_steps = 0
    start_time = time.time()

    for batch in data_iter:
        
        inputs, targets, batch_size = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        y, att_weights = model(features=inputs, 
            targets=targets, 
            max_len=max_len)
        
        y = y.permute(1, 2, 0)
        targets = targets.squeeze(2).permute(1, 0)
        
        #loss = loss_function(input=y, target=targets)
        loss = loss_func(loss_function, y, targets, att_weights, epsilon)
        loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

        l = loss.item()
        total_loss += l
        loss_log.append(l / batch_size)
        num_instances += batch_size
        num_steps += 1
    
    epoch_time = time.time() - start_time
    f_loss = total_loss / num_instances
    return f_loss, num_instances, num_steps, epoch_time, loss_log

def evaluate(model, loss_function, data_iter, max_len=MAX_LEN, epsilon=0.0005):
    """Computes loss on validation data.

    Returns:
        The loss on the dataset, a list of losses for each batch.
    """

    # set the network to evaluation mode
    model.eval()
    
    loss = 0
    loss_log = []
    num_instances = 0

    for batch in data_iter:
        i, t, batch_size = batch
        i, t = i.to(DEVICE), t.to(DEVICE)
        y, att_w = model(i, t, max_len=max_len)
        y = y.permute(1, 2, 0)
        t = t.squeeze(2).permute(1, 0)
        
        l = loss_func(loss_function, y, t, att_w, epsilon).item()
        #l = loss_function(input=y, target=t).item()

        loss += l
        loss_log.append(l / batch_size)
        num_instances += batch_size

    return (loss / num_instances), loss_log

def sample(model, data_iter, vocab, samples=1, max_len=MAX_LEN, shuffle=True):
    """Samples from the model.

    Returns:
        A list of tuples of target caption and generated caption.
    """

    # set the network to evaluation mode
    model.eval()

    if not shuffle:
        data_iter.shuffle = False
    
    samples_left = samples
    results = []

    for batch in data_iter:
        inputs, targets, batch_size = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        y, _ = model(features=inputs, targets=None, max_len=max_len)
        # y : [max_len, batch, vocab_dim]
        y = y.permute(1, 0, 2)
        _, topi = y.topk(1, dim=2)
        # topi : [batch, max_len, 1]
        topi = topi.detach().squeeze(2)
        # targets : [max_len, batch, 1]
        targets = targets.squeeze(2).permute(1, 0)
        for i in range(min(samples_left, batch_size)):
            s = ' '.join(vocab.tensor_to_sentence(topi[i]))
            t = ' '.join(vocab.tensor_to_sentence(targets[i]))
            results.append((t, s))
        samples_left -= (i + 1)
        if samples_left == 0: break
    
    if not shuffle:
        data_iter.shuffle = True

    return results

def infere(model, data_iter, vocab, max_len=MAX_LEN):
    """Perform inference with the model.

    Returns:
        A list generated caption.
    """

    # set the network to evaluation mode
    model.eval()
 
    results = []

    for batch in data_iter:
        inputs, targets, batch_size = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        y, _ = model(features=inputs, targets=None, max_len=max_len)

        # y : [max_len, batch, vocab_dim]
        y = y.permute(1, 0, 2)
        _, topi = y.topk(1, dim=2)
        
        # topi : [batch, max_len, 1]
        topi = topi.detach().squeeze(2)

        for i in range(batch_size):
            s = vocab.tensor_to_sentence(topi[i])
            results.append(s)
    
    return results

def loss_func(loss, outputs, targets, att_weigths, epsilon=0.0005):
    l = loss(input=outputs, target=targets)

    if epsilon == 0:
        return l

    penalty = 1 - torch.sum(att_weigths, dim=0)
    penalty = penalty.pow(exponent=2).sum(dim=1)
    penalty = torch.sum(epsilon * penalty)
    
    print("loss ", l.item(), " penalty ", penalty.item())

    l = l + penalty
    return l

def _write_loss_log(out_f, out_dir, log):
    with open(os.path.join(out_dir, out_f), mode='w') as f:
        for l in log:
            f.write("{0}\n".format(l))


if __name__ == "__main__":
    fire.Fire(run)
