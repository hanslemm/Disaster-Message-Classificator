import joblib
from train_classifier import evaluate_model, load_data, tokenize
from sklearn.model_selection import train_test_split

X, Y, category_names = load_data('../data/DisasterResponse.db')
model = joblib.load("classifier.pkl")

def main():
    print('Evaluating model...')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    evaluate_model(model, X_test, Y_test, category_names)


if __name__ == '__main__':
    main()