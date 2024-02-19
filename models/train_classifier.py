import pandas as pd
from sqlalchemy import create_engine
from os import path
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
from nltk.tag import pos_tag

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
download(['punkt', 'wordnet', 'stopwords'])


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
    X = disaster_resp_df.message.values
    y = disaster_resp_df.iloc[:, 4:]
    labels = y.columns  # for confusion matrix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test, labels


def tokenize(text):
    """

    :param text:
    :return:
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower()
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", clean_tok).strip()
        clean_tokens.append(clean_tok)

    clean_tokens = [tok for tok in clean_tokens if tok not in stopwords.words("english")]

    return list(filter(None, clean_tokens))


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    """

    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: x.len()).values


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """

    """

    def starting_verb(self, text):
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb).values
        return X_tagged


def build_model_pipeline():
    """

    :return:
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('txt_len', TextLengthExtractor()),
            ('verb_extract', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__nlp_pipeline__tfidf__norm': ['l1', 'l2'],
        'features__nlp_pipeline__tfidf__use_idf': [True, False],
        'clf__estimator__criterion': ['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5)

    return cv


def evaluate_model(cv, y_test, y_pred, labels):
    print(classification_report(y_test, y_pred, target_names=labels))
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


# X_train, X_test, y_train, y_test, labels = split_train_test('../data/DisasterResponse.db')
# model = build_model_pipeline()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# evaluate_model(model, y_test, y_pred, labels)
