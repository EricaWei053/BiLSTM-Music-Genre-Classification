## BiLSTM-Music-Genre-Classification 
This is a class project for class COMS6998 Fundemental Speech Recognition at Columbia. 

Author: Erica Wei

Date: 12/19/2020


## Project summary
In this project, we proposed an approach to apply efficient technique from speech recognition to extract features
and train a classifier to automatically predict music genre based on their musical characteristics.
We used librosa to extract timbral texture features, pitch feature and spectrogram features from music
to represent the audio characteristics and then apply Bidirectional Long Short-Term Memory (BiLSTM) Neural Network
to carry out classification.


## List of tools 
```
librosa
python3
pyTorch
matplotlib
seaborn
sklearn
```

## Directories needed to run
./data
./models
./results
./src
main.sh


## To quickly get results with pre-trained models
pip install -r requirements.txt
./main.sh


## To train the BiLSTM model please see below:


### How to Run
1> Please install package required in requirement.txt by
```
pip install -r requirements.txt
```

2> All the step are included in main.sh, but since the training process is time-consuming,
I just put already trained model in the folder and by running ./main.sh, it will directly
compute result for testing data.

```
./main.sh
```

3> If you want to re-train the model, please delete
``./models/ExperimentalRNN_genre6_cqt_33_batch30_model.pkl ```first and 
run ```./main.sh``` will train the model agian.
 
If you want to do the whole process again, please delete those directories and files
```
./data/_a_dev6/
./data/_a_test6/
./data/_a_train6/
./data/adev6_cqt_33_128_feature.pkl
./data/atest6_cqt_33_128_feature.pkl
./data/atrain6_cqt_33_128_feature.pkl
```
then ```./main.sh``` will do the whole process again.



## Other Details
/src/ contaisn all the scripts running whole project. Most of codes are written by Erica Wei(cw3137),
the only source codes I used from online is feature_extraction.py by http://hguimaraes.me/journal/classical-ml-mir.html,
I made some change in line 51-110 to fit our models.

/data/ contains GTZAN datasets(http://marsyas.info/downloads/datasets.html). Original data is in /genre/.
I split them into /train, /dev, and /test folders with 10 genres.
Also /_a_dev6, /_a_test6, /_a_train6 are the data with 6 genres.
And features extracted are stored in binary format ending with .pkl.

/models/ contains all the saved models trained from train.py,
most of them are just for experimental purpose and are kind of messy,
the final results are written in final_results.csv file.

/results/ contains all the historical accuracy and loss tracked during training models.


## Process Detail
- In main.sh, it runs ./src/preprocessing.py to split data into test,dev and train sets.
- Then it runs ./src/feature_extraction.py to extract all the features in each folder.
- After that it runs ./src/train.py to train BiLSTM model and test it as well, meanwhile the best model
and tracked loss & accuracy are saved. The test results will be printed on terminal instead of being stored.
- Finally, it runs ./src/postprocessing.py to check results and make plots.
