"""
Author: Erica Wei
UNI: cw3137
pre-processing.py is splitting data into train, test, dev sets.
"""
import os
import numpy as np
import subprocess as cmd
import pickle
import matplotlib.pyplot as plt
import librosa.display
d = os.path.abspath(os.getcwd())

genres_10 = {
    'metal': 0,  'classical': 1, 'hiphop': 2, 'jazz': 3,
     'pop': 4,  'reggae': 5, 'blues':6, 'country':7, 'disco':8, 'rock': 9
}

genres_6 = {
    'classical': 0, 'hiphop': 1, 'metal': 2,
     'jazz': 3, 'pop': 4, 'disco': 5,
}


src_dir = f"{d}/data/genres"


def plot_example():
    genres = {
        'metal': 0, 'classical': 1, 'hiphop': 2, 'jazz': 3,
        'pop': 4, 'reggae': 5, 'blues': 6, 'country': 7, 'disco': 8, 'rock': 9
    }
    plt.figure()
    genre_list = list(genres.keys())
    for i in range(4):
        for j in range(3):
            num = i * 3 + j
            if num > 9:
                break
            x, sr = librosa.load(f'{d}/data/example/{genre_list[num]}.00000.wav')
            plt.subplot2grid((4, 3), (i, j))
            librosa.display.waveplot(x, sr, alpha=0.8)
            plt.title(f'{genre_list[num]}', fontsize=18)
    plt.tight_layout()
    plt.show()

    plt.figure()
    for i in range(4):
        for j in range(3):
            num = i * 3 + j
            if num > 9:
                break
            y, sr = librosa.load(f'{d}/data/example/{genre_list[num]}.00000.wav')
            spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            # spect = librosa.feature.mfcc(y=y, sr=sr,n_fft=2048, hop_length=512)
            spect = librosa.power_to_db(spect, ref=np.max)

            # mfcc = librosa.feature.mfcc(y=y, sr=sr,n_fft=2048, hop_length=512)
            plt.subplot2grid((4, 3), (i, j))
            librosa.display.specshow(spect, y_axis='mel', fmax=8000, x_axis='time')
            plt.title(f'{genre_list[num]}', fontsize=24)
    plt.tight_layout()
    plt.show()


def preprocessing():
    # extract features from audio
    # feature_extraction.read_songs(d + "/data/genres")
    # exit()
    # split data to training, validation and testing parts.
    # feature_df = pd.read_csv(f"{d}/data/extracted_feature.csv")

    subscrip = ["0000" + str(x) for x in range(10)] + ["000" + str(x) for x in range(10, 100)]
    all_files = []
    training_idx, test_idx, dev_idx = [], [], []
    train_files, test_files, dev_files =  [], [], []
    # Read files from the folders
    for x, _ in genres_6.items():
        temp = []
        for sub in subscrip:
            file_name = f"{src_dir}/{x}/{x}.{sub}.wav"
            all_files.append(file_name)
            temp.append(file_name)

        test_files.extend(temp[:10])
        dev_files.extend(temp[10:20])
        train_files.extend(temp[20:])

    ''' 
    all_files = np.asarray(all_files)
    indices = [range(all_files.shape[0])]
    training_idx, test_idx, dev_idx = indices[:720], indices[720:860], indices[860:]
    train_files, test_files, dev_files = all_files[training_idx], all_files[test_idx], all_files[dev_idx]
    '''

    train_set = {}
    for fn in train_files:
        g = fn.split('/')[-1]
        g = g.split(".")[0]
        train_set[g] = train_set.get(g, 0) + 1
        cmd.run(f'cp {fn} {d}/data/_a_train6/', shell=True)
        print(f'cp {fn} {d}/data/_a_train6/')

    print("train set:")
    print(train_set)
    print(len(train_set.items()))

    test_set = {}
    for fn in test_files:
        g = fn.split('/')[-1]
        g = g.split(".")[0]

        test_set[g] = test_set.get(g, 0) + 1
        cmd.run(f'cp {fn} {d}/data/_a_test6/', shell=True)

    print("test set:")
    print(test_set)
    print(len(test_set.items()))

    dev_set = {}
    for fn in dev_files:
        g = fn.split('/')[-1]
        g = g.split(".")[0]
        dev_set[g] = dev_set.get(g, 0) + 1
        cmd.run(f'cp {fn} {d}/data/_a_dev6/', shell=True)
    print("dev set:")
    print(dev_set)
    print(len(dev_set.items()))

    print("completed ")


if __name__ == '__main__':

    print("Splitting data ")
    if not os.path.exists(f'{d}/data/_a_test6'):
        os.makedirs(f'{d}/data/_a_test6')

    if not os.path.exists(f'{d}/data/_a_dev6'):
        os.makedirs(f'{d}/data/_a_dev6')

    if not os.path.exists(f'{d}/data/_a_train6'):
        os.makedirs(f'{d}/data/_a_train6')

    if not os.path.exists(f'{d}/data/_a_train6/classical.00020.wav'):
        print("check")
        preprocessing()

    print("Done")

    # plot examples for each genre, please uncomment it if you want to see the plots.
    # plot_example()