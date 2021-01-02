
"""
Author: Erica Wei
UNI: cw3137
postprocessing.py is for getting results and visualizing.
"""
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix


d = os.path.abspath(os.getcwd())


def postprocessing(result_fn):
    result = pickle.load(open(result_fn, 'rb'))
    print(result)
    accs = result[['train_acc', 'dev_acc']]
    losss = result[['train_loss', 'dev_loss']]

    # plt.figure()
    fig1 = accs.plot().get_figure()
    plt.legend()
    plt.xlabel('epochs', fontsize=13)
    plt.title("BiSLTM accuracy over epochs ", fontsize=15)
    plt.show()
    fig1.savefig("result_acc.png")

    fig2 =losss.plot().get_figure()
    plt.legend()
    plt.xlabel('epochs', fontsize=13)
    plt.title("BiSLTM loss over epochs ", fontsize=15)
    plt.show()
    fig2.savefig("result_loss.png")


postprocessing(f"{d}/results/history_ExperimentalRNN_genre6_cqt_33_batch30.pkl")


genres_6 = {
    'classical': 0, 'hiphop': 1, 'metal': 2,
    'jazz': 3, 'pop': 4, 'disco': 5,
}

genres_6_list = list(genres_6.keys())

test_idx = [0, 4, 3, 5, 0, 4, 1, 5, 2, 0, 3, 0, 0, 5, 4, 2, 1, 3, 2, 4, 4, 1, 2, 4, 2, 1, 3, 2, 2, 4,
            1, 0, 3, 1, 0, 1, 5, 5, 4, 4, 5, 2, 0, 5, 2, 3, 4, 2, 3, 1, 1, 3, 0, 5, 3, 5, 1, 3, 5, 0]

pred_idx = [0, 4, 3, 5, 0, 4, 1, 5, 2, 0, 3, 0, 0, 5, 5, 2, 1, 3, 2, 4, 4, 1, 3, 4, 2, 1, 0, 5, 2, 3,
            2, 0, 3, 2, 0, 5, 5, 5, 1, 1, 5, 2, 0, 2, 2, 3, 4, 2, 3, 1, 1, 5, 0, 5, 0, 5, 5, 3, 5, 0]

array = confusion_matrix(test_idx, pred_idx, normalize='true')

df_cm = pd.DataFrame(array, index=[i for i in genres_6_list],
                     columns=[i for i in genres_6_list])
plt.figure(figsize=(10, 7))
fig = sn.heatmap(df_cm, annot=True).get_figure()
plt.show()
fig.savefig("confusion_matrix.png")