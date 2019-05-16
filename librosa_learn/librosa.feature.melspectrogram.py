# https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html

import librosa
import librosa.display
import numpy as np
import scipy
import scipy.signal
import scipy.fftpack


# y, sr = librosa.load(librosa.util.example_audio_file())
C = 'CantinaBand3.wav'
S = 'sample.wav'
St = 'StarWars3.wav'
y, sr = librosa.load(S)
# https://librosa.github.io/librosa/tutorial.html

librosa.feature.melspectrogram(y=y, sr=sr)
# array([[  2.891e-07,   2.548e-03, ...,   8.116e-09,   5.633e-09],
# [  1.986e-07,   1.162e-02, ...,   9.332e-08,   6.716e-09],
# ...,
# [  3.668e-09,   2.029e-08, ...,   3.208e-09,   2.864e-09],
# [  2.561e-10,   2.096e-09, ...,   7.543e-10,   6.101e-10]])

# Using a pre-computed power spectrogram

t = (librosa.stft(y)) # https://librosa.github.io/librosa/generated/librosa.core.stft.html?highlight=stft#librosa.core.stft

D = np.abs(librosa.stft(y))**2 # 위에 fourier fransform 하면 허수가 생겨서 제곱하는듯.
S = librosa.feature.melspectrogram(S=D) # https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html

# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
# https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
librosa.display.specshow(librosa.power_to_db(S,
                                             ref=np.max),
                         y_axis='mel', fmax=8000,
                         x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

plt.show()