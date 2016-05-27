import numpy as np
import tensorflow as tf
import pdb

class Dataset:

    def __init__(self, data, batch_size):
        self.data = data # TODO separate by validation and train; input and targ
        self.batch_size = batch_size
        self.nbatches = int(data.shape[0] / batch_size)

    def batch(self, i):
        if i >= self.nbatches:
            raise KeyError("Batch index too high")
        return self.data[self.batch_size*(i-1):self.batch_size*i:,]

    def shuffle(self):
        raise NameError("TODO: implement")
