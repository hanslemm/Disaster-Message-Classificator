from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import pickle
import re
import numpy as np
import pandas as pd
import nltk
import functools
import time

def timer(func):
    '''Print the runtime of the decorated function'''
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

#Update nltk resources
nltk.download(['punkt', 'wordnet', 'stopwords'])

@timer
def load_data(database_filepath):
    '''
    Load database.

    Input:
        database_filepath: path to desired database to be loaded.
    Output:
        X: dataframe with explanatory variables.
        Y: dataframe with predictive variables.
        categories: list of categories labels.
    '''
    
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Table', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    categories = df.columns.tolist()[4:]
    return X, Y, categories


def tokenize(text):
    '''
    Tokenize corpus.

    Input: corpus to be tokenized.
    Output: tokenized corpus.
    '''
    # Initialize tokens, lemmatizer and stopwords
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")

    # Normalizes and lemmatizes corpus
    lemm_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    # Remove stopwords
    clean_tokens = [t for t in lemm_tokens if t not in stop]

    return clean_tokens

@timer
def build_model():
    '''
    Model Pipeline with GridSearch optimization for parameters.

    Input: None.
    Output: classification model.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rfc', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        # 'clf__estimator__n_estimators': [50, 60, 70],
    }

    # Optimizes model parameters trough GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

@timer
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performance.

    Input:
        model: classification model 
        X_test: explanatory variables dataframe
        Y_test: predicted variables dataframe
        category_names: list of categories labels
    Output: Prints results in console.
    '''

    # Create predicted variables dataframe based on model
    y_pred = model.predict(X_test)

    # For loop incrementor
    i = 0

    # Prints classification report results for each feature
    for feature in Y_test:
        print(f'Feature: {feature}')

        # Selects each feature column and pass into classification_report
        print(classification_report(Y_test[feature], y_pred[:, i]))
        i += 1

    # Calculates global accuracy of the model
    accuracy = (y_pred == Y_test.values).mean()

    # Prints global accuracy
    print(f'The model accuracy is {accuracy}')

@timer
def save_model(model, model_filepath):
    '''
    Save classification model into a binary file (pickle).

    Input:
        model: classification model.
        model_filepath: filepath to the model.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    Main routine of the train_classifier.py file. 
    It checks the CLI command looking for 2 arguments:
        1: Actual Messages data filepath
        2: Actual Categories data filepath

    Input: none.
    Output: Loads database, print categories, build/train model and finally
        evaluates its performance.
    '''
    # Check if CLI arguments are adequate
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print(f'Loading data...\n DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)

        print(f'Categories: {category_names}')
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        try:
            evaluate_model(model, X_test, Y_test, category_names)
        except:
            print(
                'Could not evaluate model...\nPlease check your evaluate_model() code...')
        finally:
            print(f'Saving model...\n MODEL: {model_filepath}')
            save_model(model, model_filepath)
            print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
