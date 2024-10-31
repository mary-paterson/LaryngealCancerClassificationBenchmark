# preprocessors.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class MFCCPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_length=None):
        self.target_length = target_length

    def fit(self, X, y=None):
        if self.target_length is None:
            mfcc_lengths = [mfcc.shape[1] for mfcc in X]
            self.target_length = int(np.mean(mfcc_lengths))
        return self

    def transform(self, X, y=None):
        return np.array([self.clip_and_pad_2d(mfcc, self.target_length).flatten() for mfcc in X])
    
    def clip_and_pad_2d(self, array, target_length):
        if array.shape[1] > target_length:
            return array[:, :target_length]
        else:
            padding = ((0, 0), (0, target_length - array.shape[1]))
            return np.pad(array, padding, 'constant', constant_values=0)
