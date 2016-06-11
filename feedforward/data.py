import numpy as np
import tensorflow as tf
import pdb

class Dataset:

    def __init__(self, inputs, targets, batch_size):
        self.inputs = inputs
        self.targets = targets
        if batch_size > 0:
            self.batch_size = batch_size
            self.nbatches = int(inputs.shape[0] / batch_size)
        else: # run the entire dataset through
            self.batch_size = inputs.shape[0]
            self.nbatches = 1

    def batch(self, i):
        if i >= self.nbatches:
            raise KeyError("Batch index out of range")
        return self.inputs[self.batch_size*i:self.batch_size*(i+1):,], \
                self.targets[self.batch_size*i:self.batch_size*(i+1):,]

    def shuffle(self):
        raise NameError("TODO: implement")
