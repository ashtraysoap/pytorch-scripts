"""Plots graphs of losses.

The script expects a file path as command line argument.
The contents of the file should look as:

path/to/loss/log       label for the loss
path/to/another     another label
different/experiments/loss      label
...


"""


import fire
import numpy as np
import matplotlib.pyplot as plt


def main(fp):
    lines = open(fp).read().strip().split('\n')
    lines = [l.split() for l in lines]
    lines = [(l[0], ' '.join(l[1:])) for l in lines]
    
    for path, label in lines:
        log = open(path).read().strip().split('\n')
        values = {}
        values[label] = [float(l) for l in log]
        plt.plot(label, data=values, color=np.random.rand(3,))

    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    fire.Fire(main)