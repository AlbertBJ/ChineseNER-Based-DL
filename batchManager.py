__author__ = "Albert Wang"
import numpy as np


class BatchManager:
    def __init__(self, X, Y, sequence_length, shuffle=True):
        self.X = X
        self.Y = Y
        self.shuffle = shuffle
        self.start = 0
        self.count = self.X.shape[0]  # sample count
        self.sequence_length = sequence_length
        if self.shuffle:
            random_index = np.random.permutation(self.count)
            self.X = self.X[random_index]
            self.Y = self.Y[random_index]
            self.sequence_length = self.sequence_length[random_index]

    def next_batch(self, batch_size):
        end = self.start + batch_size
        b_X = self.X[self.start:end]
        b_Y = self.Y[self.start:end]
        sequence_length = self.sequence_length[self.start:end]
        if end >= self.count:
            self.start = 0  # for next epoch
        else:
            self.start = end  # for next batch
        return b_X, b_Y, sequence_length
