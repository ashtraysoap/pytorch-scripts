"""A class for loading and serving input data for the image captioning task.
"""

import os
import random

import numpy as np
import torch

class DataLoader:
    
    def __init__(
        self, 
        captions, 
        sources, 
        batch_size=1, 
        sources_prefix="", 
        vocab=None,
        max_seq_len=None,
        shuffle=True
    ):
        
        captions = list(captions)
        sources = list(sources)

        if sources_prefix != "":
            sources = [os.path.join(sources_prefix, s) for s in sources]
        
        assert len(sources) == len(captions)
        
        self.sources = sources
        self.captions = captions
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle
        self._count = len(sources)

    def __iter__(self):
        if self.shuffle == True:
            pairs = list(zip(self.sources, self.captions))
            random.shuffle(pairs)
            sources, captions = [], []
            for s, c in pairs:
                sources.append(s)
                captions.append(c)
        else:
            sources, captions = self.sources, self.captions

        counter = 0
        while counter < self._count:
            upper_bound = min(self._count, counter + self.batch_size)
            srcs = sources[counter : upper_bound]
            xs = [_load_input_data(s) for s in srcs]
            xs = [_reshape(x) for x in xs]
            xs = [torch.from_numpy(x) for x in xs]
            xs = torch.stack(xs, dim=0)
            ys = captions[counter : upper_bound]
            num_instances = len(xs)
            if self.vocab is not None:
                ys = [self.vocab.sentence_to_tensor(y, self.max_seq_len) for y in ys]
                ys = torch.stack(ys, dim=0)
                ys = ys.permute(1, 0, 2)
            yield (xs, ys, num_instances)
            counter = upper_bound


def _load_input_data(source):
    """Load the data into memory from the given path.
    """
    if source[-4:] == ".npz":
        return np.load(source)['arr_0']

    print(source)
    raise NotImplementedError("Only .npz support so far.")

def _reshape(x):
    """x is a numpy array with assumed shape (w, h, dim) or (s, dim).
    """
    if len(x.shape) == 2:
        return x
    if len(x.shape) == 3:
        dim = x.shape[2]
        return np.reshape(x, (-1, dim))
    
    raise NotImplementedError("Incorrectly shaped array given.")

if __name__ == "__main__":
    sources = open("data/feats.txt").read().strip().split('\n')
    captions = open("data/refs.txt", encoding='utf-8').read().strip().split('\n')

    dl = DataLoader(captions, sources, 2, "data/FeatsSample")
    for b in dl:
        print(b[0].size())
        print(b[1])
        print()
