import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix


def read_data(filename):
    """reads given txt file into a pandas dataframe and returns it"""
    return pd.read_csv(filename, sep='|', names=['label', 'message'])


def split_data(df):
    """
    splits data and returns training and testing sets for x and y
    return in the order of: x_train, x_test, y_train, y_test
    """
    feature, target = df['message'], df['label']
    return train_test_split(feature, target, test_size=0.5, random_state=101)


def clean_message(msg):
    """removes punctuation from given msg and returns a list of its words"""
    clean_msg = [c for c in msg if c not in string.punctuation]
    clean_msg = ''.join(clean_msg)
    clean_msg_arr = clean_msg.split(' ')
    return [x for x in clean_msg_arr if x.isalpha()]  # exclude numbers/symbols


def predict_label(msg_arr, classifier, x_train, y_train):
    """
    Returns predictions on given array of messages
    Given classifier and training data
    """
    # passes a list of steps into sklearn pipeline
    pipeline = Pipeline([
        # vectorizes tokens into numerical data using bag-of-words model
        ('vectorizer', CountVectorizer(analyzer=clean_message)),
        # computes Term Frequency - Inverse Document Frequency
        ('tfidf', TfidfTransformer()),
        # trains the model with given classifier
        ('classifier', classifier())
    ])

    # applies all pipeline steps to the given datasets and returns predictions
    pipeline.fit(x_train, y_train)
    return pipeline.predict(msg_arr)


def output_results(y_test, predictions):
    """Outputs metrics to terminal based on given test data and predictions"""
    accuracy = accuracy_score(y_test, predictions)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print(f'Accuracy: {round(accuracy * 100, 2)}%')
    print(f'True Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print(f'True Negatives: {tn}')
    print(f'True Positive Rate: {tp / (tp + fn)}')
    print(f'True Negative Rate: {tn / (tn + fp)}')
