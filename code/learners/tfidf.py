from collections import defaultdict
from datetime import datetime

import keras_metrics
import numpy
import pandas
from keras import Input, Model, Sequential
from keras.layers import Embedding, Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D, LSTM
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from keras.preprocessing.text import Tokenizer


from code.utils import log_results, save_model, get_path, encode_label


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, glove):
        self.word2vec = word2vec
        self.glove = glove
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove))])
        else:
            self.dim = 0

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
        import numpy
        return numpy.array([
            numpy.mean([self.word2vec[w] * self.word2weight[w]
                        for w in words if w in self.word2vec] or
                       [numpy.zeros(self.dim)], axis=0)
            for words in X
        ])


class NassAITfidf:
    def __init__(self, clf, data, **kwargs):
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 100)
        self.batch = kwargs.get('batch', 128)
        self.dbow = kwargs.get('dbow', 1)
        self.data = pandas.read_csv(data)
        self.result = {"type": "tfidf", "date": str(datetime.now())}

        print(self.clf)

    def show_report(self, y_test, y_pred):
        print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
        print()
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='micro')
        self.result["accuracy"] = str(accuracy)
        self.result["f1"] = f1
        print("Accuracy : {}".format(accuracy))
        print("F1 : {}".format(f1))
        return True

    def run_validation(self, clf, train_features, y_train):
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        print("Running Validation on {0}".format(clf))
        print()
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        cross_validate(clf, train_features, y_train, scoring=scoring, cv=cv)
        return True

    def mlp_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(391606,)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', keras_metrics.f1_score()])
        return model

    def cnn_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=391606, output_dim=300, input_length=391606))
        model.add(Dropout(0.2))

        model.add(Conv1D(256,
                         3,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', keras_metrics.f1_score()])
        return model

    def bilstm_model(self):
            inputs = Input(name='inputs', shape=(391606,))
            layer = Embedding(5000, 721, input_length=391606)(inputs)
            layer = LSTM(64)(layer)
            layer = Dense(256, name='FC1')(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.5)(layer)
            layer = Dense(8, name='out_layer')(layer)
            layer = Activation('sigmoid')(layer)
            model = Model(inputs=inputs, outputs=layer)
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy', keras_metrics.f1_score()])
            return model

    def train(self):
        train_data, test_data, y_train, y_test = train_test_split(self.data.clean_text, self.data.bill_class, test_size=0.2, random_state=42)
        self.result["model"] = self.clf
        if self.clf in ["svm", "mlp_sklearn", "mnb", "logreg"]:
            pipelist = [("vectorizer", TfidfVectorizer(min_df=0.2)),  ('tfidf', TfidfTransformer(use_idf=True))]

            clf_map = {"mlp_sklearn": ("mlp_sklearn", MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu')),
                       "logreg": ("logreg", LogisticRegression()),
                       "svm": ("SVC", LinearSVC()),
                       "mnb": ("mnb", MultinomialNB())}

            clf = clf_map.get(self.clf)
            pipelist.append(clf)
            pipeline = Pipeline(pipelist)
            print("Fitting data...")
            pipeline.fit(train_data, y_train)
            print()
            print("Scoring ...")
            self.run_validation(pipeline, train_data, y_train)
            y_pred = pipeline.predict(test_data)
            self.show_report(y_test, y_pred)
            log_results(self.result)
            model_file = get_path('models/tfidf_{0}.pkl'.format(self.clf))
            save_model(pipeline, model_file, True)

        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(train_data)
            x_train = tokenizer.texts_to_matrix(train_data)
            x_test = tokenizer.texts_to_matrix(test_data)
            # tfidf = TfidfVectorizer(min_df=0.2)
            # features = tfidf.fit_transform(self.data.clean_text).toarray()
            # train_features, test_features, y_train, y_test = train_test_split(features, self.data.bill_class, test_size=0.2, random_state=42)
            clf_map = {"cnn": self.cnn_model(), "bilstm": self.bilstm_model(), "mlp": self.mlp_model()}
            print("Traning Model...")
            y_train = encode_label(y_train)
            y_test = encode_label(y_test)
            model = clf_map.get(self.clf)
            model.fit(x_train, y_train,
                      batch_size=self.batch,
                      epochs=self.epoch_count,
                      validation_split=0.2)

            self.evaluate_and_report(model, x_test, y_test)
            log_results(self.result)
            model_file = get_path('models/tfidf_{0}.hd5'.format(self.clf))
            save_model(model, model_file, False)

    def evaluate_and_report(self, model, test_data, y_test):
        scores = model.evaluate(test_data, y_test, batch_size=200)
        accuracy = 100 * scores[1]
        f1 = 100 * scores[2]
        self.result["accuracy"] = accuracy
        self.result["f1"] = f1
        print("Accuracy : {}".format(accuracy))
        print("F1 : {}".format(f1))
        return True
