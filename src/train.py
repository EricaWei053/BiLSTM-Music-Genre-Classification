
"""
Author: Erica Wei
UNI: cw3137
train.py contains training and testing process for the neural network models.
"""
# Imports
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import models
from sklearn.metrics import accuracy_score

d = os.path.abspath(os.getcwd())
print(d)

genres = {
    'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
    'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9
}

# Global definitions - for NN architecture
EMBEDDING_DIM = 100
BATCH_SIZE = 30
NUM_CLASSES = 6
USE_CUDA = torch.cuda.is_available()


# Dataset

class MusicDataset(Dataset):
    def __init__(self, x_fn, y_fn):
        # data loading
        data = pickle.load(open(x_fn, 'rb'))
        y = pickle.load(open(y_fn, 'rb'))
        self.x = torch.from_numpy(data).type(torch.Tensor)
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index, :, :], self.y[index]

    def __len__(self):
        return self.n_samples


def train_model(model, loss_fn, optimizer, train_generator, dev_generator, EXP):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: models we use to train and test
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: a created optimizer we will use to update our model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """
    prev_loss = np.Infinity
    prev_acc = 0
    trained_model = model  # to hold best model
    epochs = 150
    every = 1
    train_loss_track = []
    dev_loss_track = []
    train_acc_track = []
    dev_acc_track = []

    train_num_batch = 480/30
    num_batch = 60/30
    for epoch in range(epochs):

        training_loss, training_accuracy = 0.0, 0.0
        train_gold = []
        train_pred = []
        # Set network into train set
        model.train()
        hidden = None
        for batch_x, batch_y in train_generator:
            # reset optimizer
            optimizer.zero_grad()
            # Predict outputs
            batch_x = batch_x.permute(1, 0, 2)
            outputs = model(batch_x)

            # Calculate the loss
            train_gold.extend(batch_y.cpu().detach().numpy())
            train_pred.extend(outputs.argmax(1).cpu().detach().numpy())
            loss = loss_fn(outputs, batch_y)
            # Backward and update step
            loss.backward()
            optimizer.step()

            training_loss += loss.detach().item()
        training_loss = training_loss/train_num_batch

        train_accuracy = accuracy_score(train_gold, train_pred)
        print('Epoch: ' + str(epoch) + ', Total train Loss: ' + str(training_loss)
              + ', Total train accu: ' + str(round(train_accuracy * 100, 2)) + "%")
        train_loss_track.append(training_loss)
        train_acc_track.append(train_accuracy)

        if epoch % every == 0:
            # Set network into development set
            val_gold = []
            val_pred = []
            dev_loss, dev_accuracy = 0.0, 0.0
            with torch.no_grad(): # set not gradient
                model.eval()
                # optimizer.zero_grad()

                for batch_x, batch_y in dev_generator:
                    batch_x = batch_x.permute(1, 0, 2)
                    outputs = model(batch_x)

                    # Add predictions and gold labels
                    val_gold.extend(batch_y.cpu().detach().numpy())
                    val_pred.extend(outputs.argmax(1).cpu().detach().numpy())

                    dev_loss += loss_fn(outputs.double(), batch_y.long()).detach().item()

                dev_accuracy = accuracy_score(val_gold, val_pred)
                f1 = f1_score(val_gold, val_pred, average='macro')
                dev_loss = dev_loss/num_batch
                print('Dev Epoch: ' + str(epoch) + ', Total dev Loss: ' + str(dev_loss)
                      + ', Total dev accu: ' + str(round(dev_accuracy*100, 3)) + "%")

                if dev_accuracy > prev_acc:
                    print(f"saving model... loss: {dev_loss}")
                    # prev_loss = dev_loss
                    prev_acc = dev_accuracy
                    trained_model = model
                    torch.save(trained_model, f"./models/best_model_{EXP}.pth")
            dev_loss_track.append(dev_loss)
            dev_acc_track.append(dev_accuracy)
    tracks = pd.DataFrame()
    tracks['train_loss'] = train_loss_track
    tracks['train_acc'] = train_acc_track
    tracks['dev_loss'] = dev_loss_track
    tracks['dev_acc'] = dev_acc_track

    print(tracks)
    tracks.to_csv(f"history_{EXP}.csv")
    pickle.dump(tracks, open(f"history_{EXP}.pkl", 'wb'), protocol=4)

    return trained_model


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: a model that performs 4-way emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()
    cnt = 0
    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            X_b = X_b.permute(1, 0, 2)
            y_pred = model(X_b)
            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).detach().item()
            cnt += 1

    test_accuracy = accuracy_score(gold, predicted)
    loss /= cnt
    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("Test accu: ")
    print(test_accuracy)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))

    print("True value: ")
    print(gold)

    print("Predicted: ")
    print(predicted)


def main():
    """
    Train and test neural network models for genre classification.
    """

    batch_size = 30
    feature_size = 33
    hidden_dim = 128
    n_layers = 2
    out_dim = 6

    # EXP = f"LSTM_genre6_all_38"
    #EXP = "dim256_layer3"
    EXP = f"ExperimentalRNN_genre6_cqt_33_batch30"

    print("Preprocessing all data from scratch....")
    dev_dataset = MusicDataset(f"./data/adev6_cqt_33_128_feature.pkl",
                               f"./data/adev6_cqt_33_128_target.pkl")
    train_dataset = MusicDataset(f"./data/atrain6_cqt_33_128_feature.pkl",
                                 f"./data/atrain6_cqt_33_128_target.pkl")
    test_dataset = MusicDataset(f"./data/atest6_cqt_33_128_feature.pkl",
                                f"./data/atest6_cqt_33_128_target.pkl")

    dev_generator = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
    train_generator = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_generator = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print("build model")
    # use GPU or CPU
    if USE_CUDA:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    print(EXP)


    #model = models.LSTMNet(input_dim=feature_size, hidden_dim=hidden_dim,
    #                               batch_size=batch_size, output_dim=out_dim, num_layers=n_layers)

    model = models.ExperimentalRNN(input_dim=feature_size, hidden_dim=hidden_dim,
                                    output_dim=out_dim, num_layers=n_layers)

    # learning rate
    lr = 0.001
    # loss function
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.exists(f'./models/{EXP}_model.pth'):
        trained_model = torch.load(f"./models/{EXP}_model.pth")

    else:
        model.to(device)
        trained_model = train_model(model, loss_fn, optimizer, train_generator, dev_generator, EXP)
        torch.save(trained_model, f"./models/{EXP}_model.pth")

    test_model(trained_model, loss_fn, test_generator)


if __name__ == '__main__':
    main()
