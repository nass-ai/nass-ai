import csv
import os
from collections import defaultdict

import numpy
from gensim.models.doc2vec import TaggedDocument
from pathlib import Path

from sklearn import metrics
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.models import load_model as get_model
from tensorflow.python.keras import backend as K

VECTOR_SIZE = 300
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2


def get_path(switch):
    if switch:
        return str(Path(os.path.dirname(__file__).replace('code', switch)))
    return Path(os.path.dirname(__file__))


def handle_format(text_list, train=True):
    output = []
    for index, value in enumerate(text_list):
        tag = f"train_{index}" if train else f"test_{index}"
        output.append(TaggedDocument(value.split(), [tag]))
    return output


def show_report(y_test, y_pred, classes, result):
    print(metrics.classification_report(y_test, y_pred, target_names=classes))
    print()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='micro')
    result["accuracy"] = str(accuracy)
    result["f1"] = f1
    print("Average Accuracy : {}".format(accuracy))
    print("Average F1 : {}".format(f1))
    return log_results(result)


def evaluate_and_log(model, test_data, y_test, result):
    scores = model.evaluate(test_data, y_test, batch_size=200)
    accuracy = 100 * scores[1]
    f1 = 100 * scores[2]
    result["accuracy"] = accuracy
    result["f1"] = f1
    print(accuracy, f1)
    # return log_results(result)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return numpy.array([
            numpy.mean([self.word2vec[w] for w in words if w in self.word2vec]
                       or [numpy.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return numpy.array([
            numpy.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [numpy.zeros(self.dim)], axis=0)
            for words in X
        ])


def save_model(model, filename, sklearn=True):
    if sklearn:
        _ = joblib.dump(model, filename, compress=9)
        return True
    else:
        model.save(filename)
        return True


def load_model(mode, clf):
    if clf in ["svm", "mlp_sklearn", "mnb", "logreg"]:
        filename = "models/{0}_{1}.pkl".format(mode, clf)
        return joblib.load(filename)
    elif clf == "best":
        return True
    filename = "models/{0}_{1}.hdf5".format(mode, clf)
    return get_model(filename)


def log_results(data):
    file_path = get_path('data/results.csv')
    with open(file_path, 'a') as f:
        w = csv.DictWriter(f, data.keys())
        # w.writeheader()
        w.writerow(data)
    return True


def get_vectors(model, data, get_for='train'):
    def vec(corpus_size, get_for):
            vec = numpy.zeros((corpus_size, VECTOR_SIZE))
            for i in range(0, corpus_size):
                prefix = get_for + '_' + str(i)
                vec[i] = model.docvecs[prefix]
            return vec
    vectors = handle_format(data, True)
    vectors = vec(len(vectors), get_for)
    return vectors


def encode_label(data):
    le = LabelEncoder()
    fit = le.fit_transform(data)
    output = np_utils.to_categorical(fit)
    return output


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

