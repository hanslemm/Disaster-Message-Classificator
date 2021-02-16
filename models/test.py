'''
This file has the purpose of testing the evaluation function without
generating a new classification model everytime as in train_classifier.py
'''

import joblib
from train_classifier import load_data, evaluate_model, tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

X, Y, category_names = load_data('data/DisasterResponse.db')
model = joblib.load('models/classifier.pkl')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


def main():
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)


if __name__ == '__main__':
    main()
