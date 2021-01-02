#!/bin/sh

# 1. preprocessing
python3 ./src/preprocessing.py

# 2. Feature extraction
python3 ./src/feature_extraction.py

# 3.train and test model
python3 ./src/train.py

# 4. result analysis
python3 ./src/postprocessing.py