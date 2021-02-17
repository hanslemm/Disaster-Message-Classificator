import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

#Tokenize function needed to unpickle the classification model
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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Table', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Show distribution of different category
    category = list(df.columns[4:])
    category_counts = []
    for column in category:
        category_counts.append(np.sum(df[column]))
    category_counts.sort(reverse=True)

    # extract data
    categories = df.iloc[:, 4:]
    categories_mean = categories.mean().sort_values(ascending=False)[1:11]
    categories_names = categories.columns

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Message Genres Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Message Categories Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Top 10 Message Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = [f'graph-{i}' for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
