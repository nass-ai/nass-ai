import os
from time import time
from pathlib import Path

import keras_metrics
import numpy
import pandas
from keras import Input, Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Dropout, LSTM, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


def get_path(switch):
    if switch:
        return str(Path(os.path.dirname(__file__).replace('code', switch)))
    return Path(os.path.dirname(__file__))


def encode_label(data):
    le = LabelEncoder()
    fit = le.fit_transform(data)
    output = np_utils.to_categorical(fit)
    return output


class NassAITfidf:
    def __init__(self, clf, data, **kwargs):
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 10)
        self.dbow = kwargs.get('dbow', 1)
        self.data = pandas.read_csv(data)

    @staticmethod
    def cnn_model(seq_length, vocab_size):
        print('Building Model')
        filter_sizes = [3, 4, 5]
        num_filters = 512
        inputs = Input(shape=(seq_length,))
        embedding = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_length)(inputs)
        reshape = Reshape((seq_length, 300, 1))(embedding)

        maxpool_list = []

        for i in range(2):
            conv = Conv2D(num_filters, kernel_size=(filter_sizes[0], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
            maxpool = MaxPool2D(pool_size=(seq_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv)
            maxpool_list.append(maxpool)

        concatenated_tensor = Concatenate(axis=1)(maxpool_list)
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(0.3)(flatten)
        output = Dense(units=8, activation='softmax')(dropout)
        model = Model(inputs=inputs, outputs=output)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', keras_metrics.f1_score()])
        return model

    def show_report(self, y_test, y_pred):
        print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
        print()
        print("Average Accuracy : {}".format(metrics.accuracy_score(y_test, y_pred)))
        print("Average F1 : {}".format(metrics.f1_score(y_test, y_pred, average='micro')))

    @staticmethod
    def mlp_model():
        model = Sequential()
        model.add(Dense(512, input_dim=733, init='normal', activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(8, init='normal', activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', keras_metrics.f1_score()])
        return model

    def bilstm_model(self):
        inputs = Input(name='inputs', shape=(733,))
        layer = Embedding(5000, 721, input_length=733)(inputs)
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
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        self.data['category_id'] = self.data['bill_class'].factorize()[0]
        tfidf = TfidfVectorizer(min_df=0.2)
        features = tfidf.fit_transform(self.data.clean_text).toarray()
        train_features, test_features, y_train, y_test = train_test_split(features, self.data.category_id, test_size=0.2, random_state=42)
        clf_map = {"mlp": MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu'), "mnb": MultinomialNB(), "svm": LinearSVC()}

        if self.clf in ['mnb', 'svm', 'mlp']:
            clf = clf_map.get(self.clf)
            print("Fitting data...")
            t0 = time()
            clf.fit(train_features, y_train)
            print("Done in %0.3fs" % (time() - t0))
            print()
            print("Running Validation on {0}".format(clf))
            print()
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            for epoch in range(self.epoch_count):
                accuracies = cross_validate(clf, train_features, y_train, scoring=scoring, cv=cv)
                for fold, accuracy in enumerate(accuracies):
                    print("Epoch {}".format(epoch))
                    print("Fold {0}: Accuracy={1} F1={2}".format(fold, accuracies['test_acc'][fold - 1], accuracies['test_f1_micro'][fold - 1]))
                    print()
            print("Scoring ...")
            y_pred = clf.predict(test_features)
            return self.show_report(y_test, y_pred)

        elif self.clf == "mlp_keras":
            print("Traning Model...")
            keras_model = KerasClassifier(build_fn=self.mlp_model, epochs=self.epoch_count, batch_size=self.batch, verbose=1)
            keras_model.fit(train_features, y_train)
            y_pred = keras_model.predict(test_features)
            print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
            print()
            print("Average Accuracy : {}".format(metrics.accuracy_score(y_test, y_pred)))
            print("Average F1 : {}".format(metrics.f1_score(y_test, y_pred, average='micro')))
            return True
        else:
            print(train_features.shape)
            seq_length = train_features.shape[1]
            vocab_size = len(train_features)
            print("Traning Model...")
            model = self.cnn_model(seq_length, vocab_size) if self.clf == "cnn" else self.bilstm_model()
            model.fit(train_features, y_train)
            #
            #
            # keras_model = KerasClassifier(build_fn=model, epochs=self.epoch_count, batch_size=self.batch, verbose=1)
            y_test = encode_label(y_test)
            y_train = encode_label(y_train)
            model.fit(train_features, y_train)
            y_pred = model.predict(test_features)
            self.show_report(y_test, y_pred)


d = "/Users/Olamilekan/Desktop/Machine Learning/OpenSource/nass-ai/data/clean_data.csv"
k = NassAITfidf(clf='bilstm', data=d, epoch=15).train()
