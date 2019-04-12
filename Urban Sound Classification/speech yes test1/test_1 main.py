#http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/

import glob
import os
import librosa

import librosa.display
#import librosa.logamplitude

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
#% matplotlib inline


def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X, sr = librosa.load(fp)

        raw_sounds.append(X)
    return raw_sounds


def plot_waves(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(30, 20))
    #plt.figure(figsize=(25, 60))
    #fig = plt.figure(figsize=(25,60), dpi=200)
    #fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(len(sound_names), 1, i)
        #plt.subplot(10, 1, i)
        librosa.display.waveplot(np.array(f), sr=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 1: Waveplot", x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(30, 20))
    #fig = plt.figure(figsize=(25, 60), dpi=200)
    #fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        plt.subplot(len(sound_names), 1, i)
        #plt.subplot(10, 1, i)
        specgram(np.array(f), Fs=22050)
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 2: Spectrogram", x=0.5, y=0.915, fontsize=18)
    plt.show()


def plot_log_power_specgram(sound_names, raw_sounds):
    i = 1
    fig = plt.figure(figsize=(30,20))
    #fig = plt.figure(figsize=(25, 60), dpi=300)

    #fig = plt.figure(figsize=(25, 60), dpi=900)
    for n, f in zip(sound_names, raw_sounds):
        #print(len(sound_names))
        plt.subplot(len(sound_names), 1, i)
        D = librosa.core.amplitude_to_db(np.abs(librosa.stft(f)) ** 2,ref=np.max)
        #https://librosa.github.io/librosa/changelog.html?highlight=logamplitude
        #D = librosa.logamplitude(np.abs(librosa.stft(f)) ** 2, ref_power=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.title(n.title())
        i += 1
    plt.suptitle("Figure 3: Log power spectrogram", x=0.5, y=0.915, fontsize=10)
    plt.show()


sound_file_paths = ["0a2b400e_nohash_0.wav","0a2b400e_nohash_1.wav","0a2b400e_nohash_2.wav","0a2b400e_nohash_3.wav",
                    "1b4c9b89_nohash_0.wav","1b4c9b89_nohash_1.wav","1b4c9b89_nohash_2.wav","1b4c9b89_nohash_3.wav"]

sound_names = ["man_yes_0","man_yes_1","man_yes_2","man_yes_3",
               "woman_yes_0","woman_yes_1","woman_yes_2","woman_yes_3",]

raw_sounds = load_sound_files(sound_file_paths)

#plot_waves(sound_names,raw_sounds)
#plot_specgram(sound_names,raw_sounds)
#print(librosa.show_versions())
plot_log_power_specgram(sound_names,raw_sounds)