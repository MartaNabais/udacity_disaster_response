import pickle
import re
import sys
import warnings
from os import path

import pandas as pd
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

download(['punkt', 'wordnet', 'stopwords'], quiet=True)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def read_db(db_filepath):
    """
    This function reads all columns in the database given by the filepath.
    It assumes the table name is the same as the database name with a
    _table suffix.

    :param db_filepath: relative filepath to database.
    :return:
    """
    conn = create_engine('sqlite:///' + db_filepath)
    table_name = path.basename(db_filepath.replace('.db', '') + '_table')
    disaster_resp_df = pd.read_sql_table(table_name, conn)

    return disaster_resp_df


def split_train_test(db_filepath):
    """
    This functions takes the disaster response data frame and splits into
    feature matrix, response variable and label values for training the NLP-ML pipeline.

    :param db_filepath:
    :return: a tuple with three numpy arrays X (feature matrix), y (response variable) and labels values.
    """
    disaster_resp_df = read_db(db_filepath)
    x = disaster_resp_df.message.values
    y = disaster_resp_df.iloc[:, 4:]
    labels = y.columns  # for confusion matrix
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test, labels


def tokenize(text):
    """

    :param text:
    :return:
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # removing punctuation and convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # lemmatize verbs and remove stop words
    lemmed = [lemmatizer.lemmatize(token, pos='v') for token in tokens if token not in stopwords.words("english")]

    return lemmed


def build_model_pipeline():
    """

    :return:
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__norm': ['l1', 'l2'],
        'clf__estimator__criterion': ['gini', 'entropy'],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=1, cv=5, verbose=10)

    return cv


def evaluate_model(cv, x_test, y_true, categories):
    y_pred = cv.predict(x_test)
    print(classification_report(y_true, y_pred, target_names=categories))
    print("\nBest Parameters:", cv.best_params_)


def save_model(cv, model_filepath):
    """
    This function saves the trained model in a pickle file.
    :param cv:
    :param model_filepath:
    :return:
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(cv, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        x_train, x_test, y_train, y_test, labels = split_train_test(database_filepath)

        print('Building model...')
        model = build_model_pipeline()

        print('Training model...')
        model.fit(x_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, x_test, y_test, labels)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to ' 
              'save the model to as the second argument. \n\nExample: python ' 
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
