# http://aqibsaeed.github.io/2016-09-24-urban-sound-classification-part-2/

import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# % matplotlib
# inline
plt.style.use('ggplot')


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        # start += (window_size / 2)
        start += (window_size // 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            # label = fn.split('/')[2].split('-')[1]
            label = fn.split('/')[2].split('_')[1]
            for (start, end) in windows(sound_clip, window_size):
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)

                    # logspec = librosa.logamplitude(melspec)

                    # D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f)) ** 2, ref=np.max)
                    # # https://librosa.github.io/librosa/changelog.html?highlight=logamplitude
                    # # D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)

                    logspec = librosa.core.amplitude_to_db(melspec)

                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features), np.array(labels, dtype=np.int)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


# ===========================================

parent_dir = 'Sound-Data'
tr_sub_dirs= ['fold1','fold2']
tr_features,tr_labels = extract_features(parent_dir,tr_sub_dirs)
tr_labels = one_hot_encode(tr_labels)

ts_sub_dirs= ['fold3']
ts_features,ts_labels = extract_features(parent_dir,ts_sub_dirs)
ts_labels = one_hot_encode(ts_labels)

print ("middle")

