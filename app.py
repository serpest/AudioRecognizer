import librosa
import librosa.feature as lf
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sklearn.preprocessing as skp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm

FEATURE_EXTRACTOR = {
    'chroma_stft': lf.chroma_stft,
    'melspectrogram': lf.melspectrogram,
    'mfcc': lf.mfcc,    
    'spectral_centroid': lf.spectral_centroid,
    'spectral_rolloff': lf.spectral_rolloff
}

def train_and_test_models(dataset_path, words, max_samples_per_word=50, split_ratio=0.8,
         feature_extractor_mode='melspectrogram', feature_scaling=True):
    # Data retrieving
    samples = get_sample_filenames(dataset_path, words, max_samples_per_word)
    training_samples, test_samples = split_training_test_samples(samples, split_ratio)
    # Training
    training_features = get_features(words, training_samples, feature_extractor_mode)
    random.shuffle(training_features)
    X_train = training_features[:, :-1]
    Y_train = training_features[:, -1]
    if feature_scaling:
        scaler = skp.StandardScaler()
        X_train = scaler.fit_transform(X_train)
    models = train_models(X_train, Y_train)
    # Testing
    test_features = get_features(words, test_samples, feature_extractor_mode)
    X_test = test_features[:, :-1]
    Y_test = test_features[:, -1]
    if feature_scaling:
        X_test = scaler.transform(X_test)
    return test_models(models, X_test, Y_test)

def get_sample_filenames(dataset_path, words, max_samples_per_word=-1):
    samples = dict()
    for word in words:
        directory = os.path.join(dataset_path, word)
        samples[word] = [os.path.abspath(os.path.join(directory, path))
                         for path in os.listdir(directory)]
        if max_samples_per_word > 0 and len(samples[word]) > max_samples_per_word:
            samples[word] = samples[word][:max_samples_per_word+1]
    return samples

def split_training_test_samples(samples, ratio):
    training_samples = dict()
    test_samples = dict()
    for word in samples.keys():
        split_index = int(len(samples[word]) * ratio)
        training_samples[word] = samples[word][:split_index]
        test_samples[word] = samples[word][split_index:]
    return (training_samples, test_samples)

def get_features(words, samples, mode='mfcc'):
    sample_features = []
    for word in samples.keys():
        for path in samples[word]:
            sample_features.append(get_audio_features(path, mode) + [words.index(word)])
    return np.array(sample_features)

def get_audio_features(path, mode):
    audio, sampling_rate = librosa.load(path)
    if mode not in FEATURE_EXTRACTOR:
        raise ValueError("Illegal feature extraction mode")
    features = np.mean(FEATURE_EXTRACTOR[mode](y=audio, sr=sampling_rate), axis=0)
    # See https://stackoverflow.com/questions/54221079/how-to-handle-difference-in-mfcc-feature-for-difference-audio-file
    # The parameter size=50 was chosen analyzing the specific dataset
    features = librosa.util.fix_length(features, size=50)
    return features

def train_models(X_train, Y_train):
    models = get_models()
    for i in range(len(models)):
        models[i].fit(X_train, Y_train)
    return models

def get_models():
    models = []
    models.append(KNeighborsClassifier())
    models.append(DecisionTreeClassifier(random_state=1, max_depth=10))
    models.append(RandomForestClassifier(random_state=2, max_depth=10))
    models.append(MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=10000, random_state=3))
    models.append(MLPClassifier(hidden_layer_sizes=(20, 15, 10, 10), max_iter=10000))
    models.append(SVC(random_state=5))
    return models

def test_models(models, X_test, Y_test):
    model_accuracies = []
    for model in models:
        results = model.predict(X_test)
        model_accuracies.append((str(model), skm.accuracy_score(Y_test, results)))
    return model_accuracies

def train_and_test_models_by_sample_size(dataset_path, words, sample_count_range, split_ratio=0.8,
                                         feature_extractor_mode='melspectrogram', feature_scaling=True):
    data = {}
    for i in sample_count_range:
        model_accuracies = train_and_test_models(dataset_path, words, i, split_ratio,
                                                 feature_extractor_mode, feature_scaling)
        for model, accuracy in model_accuracies:
            data.setdefault(model,[]).append(accuracy)
    for model, y_points in data.items():
        plt.plot(sample_count_range, y_points, label = model)
    plt.grid(True)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    DATASET_PATH = 'dataset'
    # WORDS = [word for word in os.listdir(DATASET_PATH)
    #          if os.path.isdir(os.path.join(DATASET_PATH, word)) and not word.startswith('_')]
    WORDS = ['down', 'go', 'left', 'bird', 'cat', 'five', 'four', 'nine', 'no']
    max_samples_per_word = 40
    sample_count_range = range(5, max_samples_per_word+1, 5)
    split_ratio = 0.8
    feature_extractor_mode = 'melspectrogram'
    feature_scaling = True
    model_accuracies = train_and_test_models(DATASET_PATH, WORDS, max_samples_per_word, split_ratio,
                                             feature_extractor_mode, feature_scaling)
    for model, accuracy in model_accuracies:
        print('{}: {:.2f}'.format(model, accuracy))
    train_and_test_models_by_sample_size(DATASET_PATH, WORDS, sample_count_range, split_ratio,
                                         feature_extractor_mode, feature_scaling)
