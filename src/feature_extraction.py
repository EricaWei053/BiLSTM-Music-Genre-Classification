"""
feature_extraction.py will extract audio features for each audio in the give directory.
"""


import os
import librosa
import pickle
import math
import numpy as np

genres_10 = {
    'metal': 0,  'classical': 1, 'hiphop': 2, 'jazz': 3,
     'pop': 4,  'reggae': 5, 'blues':6, 'country':7, 'disco':8, 'rock': 9
}


genres = {
    'classical': 0, 'hiphop': 1, 'metal': 2,
     'jazz': 3, 'pop': 4, 'disco': 5,
}


genre_list = list(genres.keys())
fix_length = 128 # we use 128 here as a length to extract feature
num_feature = 33
# chroma_feature = 'stft'
chroma_feature = 'cqt'

def get_features(y, sr, n_fft=2048, hop_length=512):
    """
    Get featuers for each audio by given window length and hop length.
    :param y: output by loading audio from librosa
    :param sr: output by loading audio from librosa
    :param n_fft: frequency
    :param hop_length: hop length.
    :return: feature array in (num_feature, fix_length) shape
    """
    # Features to concatenate in the final dictionary
    features = {}
    feature_arr = np.zeros((fix_length, num_feature), dtype=np.float64)

    # Count silence
    if 0 < len(y):
        y_sound, _ = librosa.effects.trim(y, frame_length=n_fft, hop_length=hop_length)

    if num_feature == 38:
        # Using librosa to calculate the features
        # Compute the spectral centroid.
        features['centroid'] = librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
        # Compute roll-off frequency
        features['roloff'] = librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
        # Compute zero-crossing rate of an audio time series.
        features['zcr'] = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel()
        # Compute a spectral flux onset strength envelope.
        features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
        # Compute p’th-order spectral bandwidth, p-2 by default.
        features['bandwidth'] = librosa.feature.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
        # Compute spectral flatness
        features['flatness'] = librosa.feature.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel()

        # Compute spectral contrast.
        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, hop_length=hop_length)
        for idx, v_contrast in enumerate(spectral_contrast):
            features[f'contrast_{idx}'] = v_contrast

        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

        for idx, v_chroma in enumerate(chroma):
            features['chroma_{}'.format(idx)] = v_chroma

        # MFCC treatment
        mfcc = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        for idx, v_mfcc in enumerate(mfcc):
            features['mfcc_{}'.format(idx)] = v_mfcc.ravel()

        feature_arr[:, 0] = features['centroid'].T[:fix_length]
        feature_arr[:, 1] = features['roloff'].T[:fix_length]
        feature_arr[:, 2] = features['flux'].T[:fix_length]
        feature_arr[:, 3] = features['bandwidth'].T[:fix_length]
        feature_arr[:, 4] = features['flatness'].T[:fix_length]
        feature_arr[:, 5] = features['zcr'].T[:fix_length]
        feature_arr[:, 6:18] = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).T[:fix_length]
        feature_arr[:, 18:25] = librosa.feature.spectral_contrast(y, sr=sr, hop_length=hop_length).T[:fix_length]
        feature_arr[:, 25:38] = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=13).T[:fix_length]

    elif num_feature == 33:

        feature_arr[:, 0:13] = librosa.feature.mfcc(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13).T[:fix_length]
        feature_arr[:, 13:14] = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length).T[:fix_length]
        if chroma_feature == 'stft':
            feature_arr[:, 14:26] = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).T[:fix_length]
        if chroma_feature == 'cqt':
            feature_arr[:, 14:26] = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=12)).T[:fix_length]
        feature_arr[:, 26:33] = librosa.feature.spectral_contrast(y, sr=sr, hop_length=hop_length).T[:fix_length]

    elif num_feature == 32:
        feature_arr[:, 0:13] = librosa.feature.mfcc(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13).T[
                               :fix_length]
        if chroma_feature == 'stft':
            feature_arr[:, 13:25] = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length).T[:fix_length]
        if chroma_feature == 'cqt':
            feature_arr[:, 13:25] = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=12)).T[:fix_length]
        feature_arr[:, 25:32] = librosa.feature.spectral_contrast(y, sr=sr, hop_length=hop_length).T[:fix_length]

    return feature_arr


def read_songs(src_dir, to_fn, num_data):
    """
    To get features for each audio, and return a 3-dimensional data array.
    :param src_dir: source directory
    :param to_fn: the filena,e to save feature array.
    :param num_data: number of audios want to process in the directory.
    :return: None.
    """
    # Empty array of dicts with the processed features from all files
    arr_features = []
    targets = []
    data = np.zeros((num_data, fix_length, num_feature), dtype=np.float64)
    i = 0

    for file in os.listdir(src_dir):
        if file.endswith(".wav"):
            # Read the audio file
            file_name = src_dir + "/" + file
            print("read file: ", file_name)
            targets.append(genres[file.split('.')[0]])
            signal, sr = librosa.load(file_name)
            # Append the result to the data structure
            features = get_features(signal, sr)

            data[i, :, :] = features
            i += 1

    targets = np.asarray(targets)
    print(targets.shape)
    print(data.shape)
    pickle.dump(targets, open(f'{to_fn}_target.pkl', 'wb'), protocol=4)
    pickle.dump(data, open(f'{to_fn}_feature.pkl', 'wb'), protocol=4)


def min_length(src_dir):
    """
    To find a min length over all audios,
    so that we can get a fixed length for all of them.
    :param src_dir:
    :return: min length
    """
    len_list = []
    for g in genres.keys():
        genre_dir = src_dir + f"/{g}"
        print(genre_dir)
        for root, subdirs, files in os.walk(genre_dir):
            for file in files:
                file_name = genre_dir + "/" + file
                print("Loading " + str(file_name))
                y, sr = librosa.load(file_name)
                len_list.append(math.ceil(len(y) / 512))
    print(min(len_list)) # = 1290
    return min(len_list)


d = os.path.abspath(os.getcwd())
print(d)
# fix_length = min_length(f"{d}/data/genres")
if not os.path.exists(f"{d}/data/atest6_{chroma_feature}_{num_feature}_{fix_length}_feature.pkl"):
    read_songs(f"{d}/data/_a_test6", f"{d}/data/atest6_{chroma_feature}_{num_feature}_{fix_length}", 60)
    print("test set feature extraction finished")
if not os.path.exists(f"{d}/data/adev6_{chroma_feature}_{num_feature}_{fix_length}_feature.pkl"):
    read_songs(f"{d}/data/_a_dev6", f"{d}/data/adev6_{chroma_feature}_{num_feature}_{fix_length}", 60)
    print("dev set feature extraction finished")
if not os.path.exists(f"{d}/data/atrain6_{chroma_feature}_{num_feature}_{fix_length}_feature.pkl"):
    read_songs(f"{d}/data/_a_train6", f"{d}/data/atrain6_{chroma_feature}_{num_feature}_{fix_length}", 480)
    print("train set feature extraction finished")

print("feature extraction done.")