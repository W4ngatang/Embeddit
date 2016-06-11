import numpy as np
import tensorflow as tf
import pdb

class Dataset:

    def __init__(self, batch_size, inputs, targets=None):
        self.inputs = inputs # input should be 1 x #elts array
        self.targets = targets # target should be same as inputs but shifted by one
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.block_size = batch_size * seq_len # the number of indices in an entire batch
        self.nbatches = int(inputs.shape[0] / block_size)

    def batch(self, i):
        if i >= self.nbatches:
            raise KeyError("Batch index out of range")
        if self.targets is not None:
            return self.inputs[self.block_size*i:self.block_size*(i+1)].\
                        reshape((batch_size, seq_len)),\
                    self.targets[self.block_size*i:self.block_size*(i+1)].\
                        reshape((batch_size, seq_len))
        else: # only if test
            return self.inputs[self.block_size*i:self.block_size*(i+1)].\
                        reshape((batch_size, seq_len))

    # TODO 
    def shuffle(self):
        return
        
