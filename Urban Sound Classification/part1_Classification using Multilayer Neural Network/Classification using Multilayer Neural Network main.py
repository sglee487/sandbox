import glob
import os
import librosa

import librosa.display
#import librosa.logamplitude

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

training_epochs = 50
n_dim = tr_features.shape[1]
n_classes = 10
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

