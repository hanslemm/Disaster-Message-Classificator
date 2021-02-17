'''
This file has the purpose of testing the evaluation function without
generating a new classification model everytime as in train_classifier.py
'''

import joblib
from data.process_data import load_data
from models.train_classifier import load_database, evaluate_model, tokenize
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

load_data('data/disaster_messages.csv','data/disaster_categories.csv')
X, Y, category_names = load_database('data/DisasterResponse.db')
model = joblib.load('models/classifier.pkl')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


def main():
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)


if __name__ == '__main__':
    main()
