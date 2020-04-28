import os
import sys
import time

import fire
import torch
import torch.nn as nn

from data_loader import DataLoader
from prepro import normalize_strings, filter_inputs
from vocab import Vocab

from attndec_batched import Network

MAX_LEN = 15
HIDDEN_DIM = 512
EPOCHS = 100
BATCH_SIZE = 4
CLIP_VAL = 1

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE:\t", DEVICE)

def run(train_feats, 
    train_caps, 
    val_feats, 
    val_caps, 
    train_prefix="",
    val_prefix="",
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    max_seq_len=MAX_LEN,
    hidden_dim=HIDDEN_DIM,
    clip_val=CLIP_VAL,
    out_dir="Pytorch_Exp_Out"):
    
    print("EXPERIMENT START ", time.asctime())

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # 1. Load the data

    train_captions = open(train_caps, mode='r', encoding='utf-8') \
        .read().strip().split('\n')

    val_captions = open(val_caps, mode='r', encoding='utf-8') \
        .read().strip().split('\n')

    train_features = open(train_feats, mode='r').read().strip().split('\n')
    train_features = [os.path.join(train_prefix, z) for z in train_features]

    val_features = open(val_feats, mode='r').read().strip().split('\n')
    val_features = [os.path.join(val_prefix, z) for z in val_features]

    assert len(train_captions) == len(train_features)
    assert len(val_captions) == len(val_features)

    # 2. Preprocess the data

    train_captions = normalize_strings(train_captions)
    val_captions = normalize_strings(val_captions)

    train_data = list(zip(train_captions, train_features))
    val_data = list(zip(val_captions, val_features))

    train_data = filter_inputs(train_data)
    val_data = filter_inputs(val_data)

    vocab = Vocab()
    vocab.build_vocab(map(lambda x: x[0], train_data), max_size=10000)
    vocab.save(path=os.path.join(out_dir, 'vocab.txt'))

    print("Total training instances: ", len(train_data))
    print("Total validation instances: ", len(val_data))
    print("Vocabulary size: ", vocab.n_words)

    # 3. Initialize the network, optimizer & loss function

    net = Network(hidden_size=hidden_dim, output_size=vocab.n_words,
        sos_token=0, eos_token=1, pad_token=2)
    net.to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters())
    loss_function = nn.NLLLoss()

    # 4. Train

    prev_val_l = sys.maxsize
    total_instances = 0
    total_steps = 0
    train_loss_log = []
    val_loss_log = []

    train_data = DataLoader(captions=map(lambda x: x[0], train_data),
        sources=map(lambda x: x[1], train_data), batch_size=batch_size, 
        vocab=vocab, max_seq_len=max_seq_len)
    val_data = DataLoader(captions=map(lambda x: x[0], val_data),
        sources=map(lambda x: x[1], val_data), batch_size=batch_size, 
        vocab=vocab, max_seq_len=max_seq_len)

    training_start_time = time.time()

    for e in range(1, epochs + 1):
        print("Epoch ", e)

        # train one epoch
        train_l, inst, steps, t = train_epoch(model=net, loss_function=loss_function,
            optimizer=optimizer, data_iter=train_data, max_len=max_seq_len, clip_val=clip_val)
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
        print()


        # evaluate
        val_l = evaluate(model=net, loss_function=loss_function, 
            data_iter=val_data, max_len=max_seq_len)
        print("Validation loss: ", val_l)
        if val_l < prev_val_l:
            torch.save(net.state_dict(), os.path.join(out_dir, 'net.pt'))
        val_loss_log.append(val_l)


        #sample model
        samples = sample(net, train_data, vocab, samples=3, max_len=max_seq_len)
        for t, s in samples:
            print("Target:\t", t)
            print("Predicted:\t", s)
            print()

    tot_time = time.time() - training_start_time
    hours = tot_time // 3600
    mins = (tot_time % 3600) // 60
    secs = (tot_time % 60)
    print("Total training time:\t{0}:{1}:{2}".format(hours, mins, secs))
    print("Total training instances:\t", total_instances)
    print("Total training steps:\t", total_steps)
    print()

    with open(os.path.join(out_dir, "train_loss_log.txt"), mode='w') as tl_log:
        for l in train_loss_log:
            tl_log.write("{0}\n".format(l))
    with open(os.path.join(out_dir, "val_loss_log.txt"), mode='w') as vl_log:
        for l in val_loss_log:
            vl_log.write("{0}\n".format(l))

    print("EXPERIMENT END ", time.asctime())

def train_epoch(model, loss_function, optimizer, data_iter, max_len=MAX_LEN, clip_val=CLIP_VAL):
    model.train()

    total_loss = 0
    num_instances = 0
    num_steps = 0
    start_time = time.time()

    for batch in data_iter:
        
        inputs, targets, batch_size = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        y = model(features=inputs, 
            targets=targets, 
            max_len=max_len)
        
        y = y.view(batch_size, -1, max_len)
        targets = targets.view(batch_size, max_len)
        
        loss = loss_function(input=y, target=targets)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()

        total_loss += loss.item()
        num_instances += batch_size
        num_steps += 1
    
    epoch_time = time.time() - start_time
    f_loss = total_loss / num_instances
    return f_loss, num_instances, num_steps, epoch_time

def evaluate(model, loss_function, data_iter, max_len=MAX_LEN):
    model.eval()
    
    loss = 0
    num_instances = 0

    for batch in data_iter:
        i, t, batch_size = batch
        i, t = i.to(DEVICE), t.to(DEVICE)
        y = model(i, t, max_len=max_len)
        y = y.view(batch_size, -1, max_len)
        t = t.view(batch_size, max_len)
        loss += loss_function(input=y, target=t).item()
        num_instances += batch_size

    return loss / num_instances

def sample(model, data_iter, vocab, samples=1, max_len=MAX_LEN):
    model.eval()
    data_iter.shuffle = False
    samples_left = samples
    results = []

    for batch in data_iter:
        inputs, targets, batch_size = batch
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        y = model(inputs, None, max_len)
        # y : [max_len, batch, vocab_dim]
        y = y.view(batch_size, max_len, -1)
        _, topi = y.topk(1, dim=2)
        # topi : [batch, max_len, 1]
        topi = topi.detach().view(batch_size, max_len)
        # targets : [max_len, batch, 1]
        targets = targets.view(batch_size, max_len)
        for i in range(min(samples_left, batch_size)):
            s = vocab.tensor_to_sentence(topi[i])
            t = vocab.tensor_to_sentence(targets[i])
            results.append((t, s))
        samples_left -= (i + 1)
        if samples_left == 0: break
    
    data_iter.shuffle = True
    return results



if __name__ == "__main__":
    fire.Fire(run)
