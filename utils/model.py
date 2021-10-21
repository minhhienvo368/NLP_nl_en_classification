
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from utils/my_functions import *


if __name__ == '__main__':
    try:
        df_msg = read_data('raw_text.txt')

        # splits data between train and test
        x_train, x_test, y_train, y_test = split_data(df_msg)

        # predicts with Naive Bayes classifier and output results to terminal
        nb_pred = predict_label(x_test, MultinomialNB, x_train, y_train)
        print('Results for Naive Bayes classifier:')
        output_results(y_test, nb_pred)

        print()

        # predicts with Random Forest classifier and output results to terminal
        rf_pred = predict_label(x_test, RandomForestClassifier,
                                x_train, y_train)
        print('Results for Random Forest classifier:')
        output_results(y_test, rf_pred)

    except Exception as e:
        print(f'could not execute script: {e}')
