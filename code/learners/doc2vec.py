from pathlib import Path

import numpy
import pandas
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from keras.layers import Bidirectional
from keras import Input, Model
from keras.initializers import Constant
from keras.layers import Embedding, Flatten, Dense, Dropout, LSTM, GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from code.utils import get_path, encode_label, get_vectors, f1, evaluate_and_log, show_report
from code.utils import MeanEmbeddingVectorizer


class NassAIDoc2Vec:
    def __init__(self, clf, data, **kwargs):
        print(kwargs)
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 10)
        self.dbow = kwargs.get('dbow', 1)
        self.use_glove = kwargs.get('use_glove', True)
        self.data = pandas.read_csv(data)
        self.nass_embedding_path = get_path('models/doc2vec/nassai_dbow_doc2vec.vec') if self.dbow else get_path('models/doc2vec/nassai_dm_doc2vec.vec')
        self.max_sequence_length = 300
        self.max_num_words = 50000
        self.embedding_dim = 300
        self.validation_split = 0.2
        self.num_words = None
        self.embedding_matrix = None
        self.result = {"type": "word2vec"}

    def cnn_model(self):
        print('Building Model')
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(8, activation='softmax')(x)

        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', f1])
        return model

    def bilstm_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))(embedded_sequences)
        x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(8, activation='softmax')(x)
        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', f1])
        model._name = 'bi_model'
        return model

    def mlp_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dense(512, activation='relu')(embedded_sequences)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu')(x)
        predictions = Dense(8, activation='softmax')(x)

        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', f1])
        return model

    def train(self):
        print('Indexing word vectors.')

        texts = self.data.clean_text
        print('Found %s texts.' % len(texts))
        labels = self.data.bill_class

        embeddings_index = {}

        if not Path(self.nass_embedding_path).is_file():
            tagged = [TaggedDocument(words=word_tokenize(_d.lower()),
                                     tags=[str(i)]) for i, _d in enumerate(texts)]
            model = Doc2Vec(vector_size=100,
                            window=5,
                            alpha=.025,
                            min_alpha=0.00025,
                            min_count=2,
                            dm=1,
                            workers=8)
            model.build_vocab(tagged)
            print("Building Model")
            epochs = range(10)
            for epoch in epochs:
                print(f'Epoch {epoch+1}')
                model.train(tagged,
                            total_examples=model.corpus_count,
                            epochs=model.epochs)
                # decrease the learning rate
                model.alpha -= 0.00025
                # fix the learning rate, no decay
                model.min_alpha = model.alpha
        else:

            print("Model Exists. Loading")
            model = Doc2Vec.load(self.nass_embedding_path)
        tags = model.docvecs.index2entity
        for tag in tags:
                embeddings_index[tag] = model.docvecs[tag]

        print('Found %s word vectors.' % len(embeddings_index))
        print('Processing text dataset')

        tokenizer = Tokenizer(num_words=self.max_num_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        data = pad_sequences(sequences, maxlen=self.max_sequence_length)

        print('Preparing embedding matrix.')

        self.num_words = min(self.max_num_words, len(word_index)) + 1
        self.embedding_matrix = numpy.zeros((self.num_words, self.embedding_dim))
        for word, i in word_index.items():
            if i > self.max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

        if self.clf in ["cnn", "bilstm", "mlp"]:
            self.result["model"] = self.clf
            labels = encode_label(labels)

            train_data, test_data, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

            train_vectors = get_vectors(model, train_data)
            test_vectors = get_vectors(model, test_data, 'test')

            clf_map = {"cnn": self.cnn_model(), "bilstm": self.bilstm_model(), "mlp": self.mlp_model()}

            clf_model = clf_map.get(self.clf)

            clf_model.fit(train_vectors, y_train,
                          batch_size=self.batch,
                          epochs=self.epoch_count,
                          validation_split=0.2)

            return evaluate_and_log(clf_model, test_vectors, y_test, self.result)

        elif self.clf in ["svm", "mlp_sk", "mnb", 'logreg']:
            train_data, test_data, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            train_vectors = get_vectors(model, train_data)
            test_vectors = get_vectors(model, test_data, 'test')
            pipelist = [("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index, self.embedding_dim))]

            clf_map = {"mlp_sk": ("mlp_sk", MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu')),
                       "logreg": ("logreg", LogisticRegression()), "svm": ("SVC", LinearSVC())}
            clf = clf_map.get(self.clf)
            pipelist.append(clf)
            pipeline = Pipeline(pipelist)
            print("Fitting data...")
            pipeline.fit(train_vectors, y_train)
            print()
            print("Scoring ...")
            y_pred = pipeline.predict(test_vectors)
            return show_report(y_test, y_pred, self.data['bill_class'].unique(), self.result)

    def run_validation(self, clf, train, y_train):
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        print("Running Validation on {0}".format(clf))
        print()
        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        for epoch in range(self.epoch_count):
            accuracies = cross_validate(clf, train, y_train, scoring=scoring, cv=cv)
            print("Epoch {}".format(epoch))
            for fold, accuracy in enumerate(accuracies):
                print("Fold {0}: Accuracy={1} F1={2}".format(fold, accuracies['test_acc'][fold - 1], accuracies['test_f1_micro'][fold - 1]))
                print()
        return True
