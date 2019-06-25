from time import time
from pathlib import Path

import keras_metrics
import numpy
import pandas
from gensim.models import Doc2Vec
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from code import get_path
from code.learners import encode_label


class NassAITfidf:
    def __init__(self, clf, data, **kwargs):
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 10)
        self.dbow = kwargs.get('dbow', 1)
        self.data = pandas.read_csv(data)

    def cnn_model(self, seq_length, vocab_size):
        print('Building Model')
        filter_sizes = [3, 4, 5]
        num_filters = 512
        inputs = Input(shape=(seq_length,))
        embedding = Embedding(input_dim=vocab_size, output_dim=300, input_length=seq_length)(inputs)
        reshape = Reshape((seq_length, 300, 1))(embedding)
        conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
        conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], 300), padding='valid', kernel_initializer='normal', activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(seq_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(seq_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(seq_length - filter_sizes[2] + 1, 1), strides=(1, 1), padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(0.3)(flatten)
        output = Dense(units=8, activation='softmax')(dropout)

        model = Model(inputs=inputs, outputs=output)

        adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', keras_metrics.f1_score()])
        return model

    def bilstm_model(self):
        pass

    def train(self):
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        self.data['category_id'] = self.data['bill_class'].factorize()[0]
        tfidf = TfidfVectorizer()
        model_path = get_path('data') + '/tfidf.npz'
        if Path(model_path).is_file():
            print("tfidf file exist. Loading ..")
            features = numpy.load(model_path)
        else:
            print("tfidf file does not exist. Creating ..")
            features = tfidf.fit_transform(self.data.clean_text).toarray()
            numpy.savez_compressed(model_path, features)
            print("Saved TFID ..")
        train_features, test_features, y_train, y_test = train_test_split(features, self.data.category_id, test_size=0.2, random_state=42)
        clf_map = {"mlp": MLPClassifier(alpha=1, max_iter=1000), "mnb": MultinomialNB(), "svm": LinearSVC()}
        if self.clf in ['mnb', 'svm', 'mlp']:
            clf = clf_map.get(self.clf)
            print("Fitting data...")
            t0 = time()
            clf.fit(train_features, y_train)
            print("done in %0.3fs" % (time() - t0))
            print()
            print("Running Validation on {0}".format(clf))
            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
            for epoch in range(self.epoch_count):
                accuracies = cross_validate(clf, train_features, y_train, scoring=scoring, cv=cv)
                for fold, accuracy in enumerate(accuracies):
                    print("Epoch {0}: Accuracy={1} F1={2}".format(fold, accuracies['test_acc'][fold - 1], accuracies['test_f1_micro'][fold - 1]))

            print("Scoring ...")
            y_pred = clf.predict(test_features)
            from sklearn import metrics
            print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
            return True
        else:
            model_path = get_path('models/doc2vec')
            model_path = model_path + '/dbow_doc2vec.vec' if self.dbow else model_path + '/dm_doc2vec.vec'
            print("Loading model")
            doc2vec_model = Doc2Vec.load(model_path)
            print("Model Loaded")
            seq_length = train_features.shape[1]
            vocab_size = len(doc2vec_model.docvecs)
            model = self.cnn_model(seq_length, vocab_size) if self.clf == "cnn" else self.bilstm_model()
            print("Traning Model...")
            checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
            print(train_features.shape)
            print(y_train.shape)
            y_train = encode_label(y_train)
            y_test = encode_label(y_test)
            fitted_model = model.fit(train_features, y_train, batch_size=self.batch, epochs=self.epoch_count, verbose=1, callbacks=[checkpoint], validation_split=0.2)
            print("Training Accuracy: %.2f%% \nValidation Accuracy: %.2f%%" % (100 * fitted_model.history['acc'][-1], 100 * fitted_model.history['val_acc'][-1]))
            print("Training F1: %.2f%% \nValidation F1: %.2f%%" % (100 * fitted_model.history['f1_score'][-1], 100 * fitted_model.history['val_f1_score'][-1]))
            scores = model.evaluate(test_features, y_test, batch_size=200)
            print("Test Accuracy: %.2f%%" % (100 * scores[1]))
            print("Test F1: %.2f%%" % (100 * scores[2]))

