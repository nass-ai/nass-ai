import numpy
import pandas
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from keras import Input, Model
from keras.initializers import Constant
from keras.layers import Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dense, Dropout, Activation, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from keras.preprocessing.text import Tokenizer
import keras_metrics
from keras_preprocessing.sequence import pad_sequences
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from code.utils import get_path, encode_label
from code.utils import MeanEmbeddingVectorizer, log_results


class NassAIWord2Vec:
    def __init__(self, clf, data, **kwargs):
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 10)
        self.cv = kwargs.get('cv', 5)
        self.cbow = kwargs.get('cbow', 1)
        self.use_glove = 1
        self.data = pandas.read_csv(data)
        self.glove_path = get_path('models/glove/glove.6B.300d.txt')
        self.nass_embedding_path = get_path('mdoels/word2vec/nassai_word2vec.vec')
        self.encoding = "utf-8"
        self.word2vecmodel = None
        self.max_sequence_length = 1000
        self.max_num_words = 20000
        self.embedding_dim = 300
        self.validation_split = 0.2
        self.num_words = None
        self.embedding_matrix = None
        self.result = {"type": "word2vec"}

    def show_report(self, y_test, y_pred):
        print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
        print()
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='micro')
        self.result["accuracy"] = str(accuracy)
        self.result["f1"] = f1
        print("Average Accuracy : {}".format(accuracy))
        print("Average F1 : {}".format(f1))

    def evaluate_and_report(self, model, test_data, y_test):
        scores = model.evaluate(test_data, y_test, batch_size=200)
        accuracy = 100 * scores[1]
        f1 = 100 * scores[2]
        self.result["accuracy"] = accuracy
        self.result["f1"] = f1
        return True

    def bilstm_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
        x = Dropout(0.2)(x)
        x = Dense(256, name='Dense_1', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, name='Dense_2', activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(8, name='output', activation='softmax')(x)
        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', keras_metrics.f1_score()])
        return model

    def cnn_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)

        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Conv1D(128, 5, activation='relu', name='conv1D_1')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu', name='conv1D_2')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu', name='conv1D_3')(x)
        x = GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu', name='output')(x)
        predictions = Dense(8, activation='softmax')(x)

        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', keras_metrics.f1_score()])
        return model

    def mlp_model(self):
        embedding_layer = Embedding(self.num_words,
                                    self.embedding_dim,
                                    embeddings_initializer=Constant(self.embedding_matrix),
                                    input_length=self.max_sequence_length,
                                    trainable=False)
        sequence_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Dense(512, activation='relu', name='mlp_dense1')(embedded_sequences)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', name='mlp_dense2')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu', name='mlp_dense3')(x)
        x = Dropout(0.2)(x)
        x = Dense(8, activation='relu', name='output')(x)
        predictions = Dense(8, activation='softmax')(x)

        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', keras_metrics.f1_score()])
        return model

    def train(self):
        print('Indexing word vectors.')

        embeddings_index = {}

        texts = self.data.apply(lambda r: simple_preprocess(r['clean_text'], min_len=2), axis=1)
        print('Found %s texts.' % len(texts))
        labels = self.data.bill_class

        if self.use_glove:
            with open(self.glove_path) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = numpy.fromstring(coefs, 'f', sep=' ')
                    embeddings_index[word] = coefs

        else:
            print("Building word2vec embedding")
            model = Word2Vec(texts, size=self.embedding_dim, window=5, min_count=5, workers=2, hs=1, sg=self.cbow, negative=5, alpha=0.065, min_alpha=0.065)
            embeddings_index = dict(zip(model.wv.index2word, model.wv.syn0))

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

        self.result["model"] = self.clf

        if self.clf in ["cnn", "bilstm", "mlp"]:
            labels = encode_label(labels)
            train_data, test_data, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

            clf_map = {"cnn": self.cnn_model(), "bilstm": self.bilstm_model(), "mlp": self.mlp_model()}

            model = clf_map.get(self.clf)

            model.fit(train_data, y_train,
                      batch_size=self.batch,
                      epochs=self.epoch_count,
                      validation_split=0.2)

            self.evaluate_and_report(model, test_data, y_test)
            return log_results(self.result)

        else:
            train_data, test_data, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            pipelist = [("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index))]

            clf_map = {"mlp_sk": ("mlp_sk", MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu')),
                       "logreg": ("logreg", LogisticRegression()), "svm": ("SVC", LinearSVC())}
            clf = clf_map.get(self.clf)
            pipelist.append(clf)
            pipeline = Pipeline(pipelist)
            print("Fitting data...")
            pipeline.fit(train_data, y_train)
            print()
            print("Scoring ...")
            self.run_validation(pipeline, train_data, test_data)
            y_pred = pipeline.predict(test_data)
            self.show_report(y_test, y_pred)
            return log_results(self.result)

    def run_validation(self, clf, train, y_train):
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        print("Running Validation on {0}".format(clf))
        print()
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=42)
        accuracies = cross_validate(clf, train, y_train, scoring=scoring, cv=cv)
        print(accuracies)
        return True
