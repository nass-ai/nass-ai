from datetime import datetime

import keras_metrics
import pandas
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, LSTM, Bidirectional
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from code.utils import log_results, save_model, get_path, f1


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

    def mlp_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(739,)))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', f1])
        return model

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=500,
                        activation='relu', input_shape=(733,)))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))
        model.compile(
            optimizer=Adam(0.2),
            loss='categorical_crossentropy',
            metrics=['accuracy', f1])
        return model

    def cnn_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=739, output_dim=300, input_length=739))
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
        model = Sequential()
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(1886,))))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(8, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', keras_metrics.f1_score()])
        return model

    def train(self):
        train_data, test_data, y_train, y_test = train_test_split(self.data.clean_text, self.data.bill_class, test_size=0.2, random_state=42)
        self.result["model"] = self.clf
        pipelist = [("vectorizer", TfidfVectorizer(min_df=0.2)), ('tfidf', TfidfTransformer(use_idf=True))]
        if self.clf in ["svm", "mlp_sklearn", "mnb", "logreg", 'random_forest', 'svm_linear']:
            sklearn_clf_map = {"mlp_sklearn": ("mlp_sklearn", MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu')),
                               "logreg": ("logreg", LogisticRegression()),
                               "svm_linear": ("Linear_SVC", LinearSVC()),
                               "svm": ("SVC", SVC()),
                               "random_forest": ("Random_Forest", RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)),
                               "mnb": ("mnb", MultinomialNB())}

            clf = sklearn_clf_map.get(self.clf)
            save_path = 'models/tfidf_{0}.pkl'.format(self.clf)
        else:
            keras_clf_map = {"cnn": ("cnn", KerasClassifier(build_fn=self.cnn_model, epochs=self.epoch_count, batch_size=self.batch, verbose=1, validation_split=0.2)),
                             "bilstm": ("bilstm", KerasClassifier(build_fn=self.bilstm_model, epochs=self.epoch_count, batch_size=self.batch, verbose=1, validation_split=0.2)),
                             "mlp": ("mlp", KerasClassifier(build_fn=self.mlp_model, epochs=self.epoch_count, batch_size=self.batch, verbose=1, validation_split=0.2))
                             }
            clf = keras_clf_map.get(self.clf)
            save_path = 'models/tfidf_{0}.hd5'.format(self.clf)
        pipelist.append(clf)
        print(pipelist)
        pipeline = Pipeline(pipelist)
        pipeline.fit(train_data, y_train)
        print()
        print("Scoring ...")
        y_pred = pipeline.predict(test_data)
        self.show_report(y_test, y_pred)
        log_results(self.result)
        model_file = get_path(save_path)
        save_model(pipeline, model_file, True)

    def evaluate_and_report(self, model, test_data, y_test):
        scores = model.evaluate(test_data, y_test, batch_size=200)
        accuracy = 100 * scores[1]
        f1 = 100 * scores[2]
        self.result["accuracy"] = accuracy
        self.result["f1"] = f1
        return True
