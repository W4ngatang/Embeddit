import numpy as np
import tensorflow as tf
import pdb

class Dataset:

    def __init__(self, inputs, targets=None):
        self.inputs = inputs 
        self.targets = targets 
        self.seq_len = inputs.shape[2]
        self.batch_size = inputs.shape[1]
        self.nbatches = inputs.shape[0]

    def batch(self, i):
        return self.inputs[i], self.targets[i] if self.targets is not None \
            else self.inputs[i]

    # TODO 
    def shuffle(self):
        return
        
