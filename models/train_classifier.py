import sys, pickle, re

import numpy as np

import pandas as pd

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Table', engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories = df.columns.tolist()[4:]
    return X,Y,categories


def tokenize(text):
    
    #Initialize tokens, lemmatizer and stopwords
    tokens=word_tokenize(text)
    lemmatizer=WordNetLemmatizer()
    stop = stopwords.words("english")
    
    lemm_tokens= [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    clean_tokens = [t for t in lemm_tokens if t not in stop]
    
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [50, 60, 70],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    i = 0
    for col in Y_test:
        print(f'Feature {i+1}: {col}')
        print(classification_report(Y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test.values).mean()
    print(f'The model accuracy is {accuracy}')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        print(f'Categories: {category_names}')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print(Y_test.values)
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        try:
            evaluate_model(model, X_test, Y_test, category_names)
        except:
            print('Could not evaluate model...\nPlease check your evaluate_model() code...')
        finally:
            print(f'Saving model...\n MODEL: {model_filepath}')
            save_model(model, model_filepath)
            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()