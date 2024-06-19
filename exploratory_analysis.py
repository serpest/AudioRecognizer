import librosa
import librosa.feature as lf
import matplotlib.pyplot as plt
import numpy as np

import app

DATASET_PATH = 'dataset'
WORDS = ['down', 'left', 'no']
SAMPLES = app.get_sample_filenames(DATASET_PATH, WORDS, 4)

assert len(app.FEATURE_EXTRACTORS) == 4

# Audio visualization
audio, sampling_rate = librosa.load(SAMPLES[WORDS[0]][0])
plt.subplot(2, 1, 1)
plt.title('waveplot')
librosa.display.waveshow(audio, sr=sampling_rate)
plt.subplot(2, 1, 2)
plt.title('chroma_stft')
librosa.display.specshow(lf.chroma_stft(y=audio, sr=sampling_rate), x_axis='time', y_axis='log')
plt.get_current_fig_manager().set_window_title('Audio visualization')
plt.show()

# Features visualization with different words
audios = [librosa.load(SAMPLES[word][0]) for word in WORDS]
for i, mode in enumerate(app.FEATURE_EXTRACTORS.keys()):
    plt.subplot(2, 2, i+1)
    plt.title(mode)
    for audio, sampling_rate in audios:
        features = np.mean(app.FEATURE_EXTRACTORS[mode](y=audio, sr=sampling_rate), axis=0)
        plt.plot(features)
plt.get_current_fig_manager().set_window_title('Features visualization with different words')
plt.show()

# Features visualization with same word
audios = [librosa.load(SAMPLES[WORDS[0]][i]) for i in range(4)]
assert len(app.FEATURE_EXTRACTORS) == 4
for i, mode in enumerate(app.FEATURE_EXTRACTORS.keys()):
    plt.subplot(2, 2, i+1)
    plt.title(mode)
    for audio, sampling_rate in audios:
        features = np.mean(app.FEATURE_EXTRACTORS[mode](y=audio, sr=sampling_rate), axis=0)
        plt.plot(features)
plt.get_current_fig_manager().set_window_title('Features visualization with same word')
plt.show()
