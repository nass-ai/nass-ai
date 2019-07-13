import numpy
import pandas
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tensorflow.python.keras import Input, Model, Sequential
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.layers import Embedding, Dense, Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from code.utils import get_path, encode_label, f1, show_report, evaluate_and_log, evaluate_and_log
from code.utils import MeanEmbeddingVectorizer


class NassAIWord2Vec:
    def __init__(self, clf, data, **kwargs):
        print(kwargs)
        self.clf = clf
        self.epoch_count = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 10)
        self.cv = kwargs.get('cv', 5)
        self.cbow = kwargs.get('cbow', 1)
        self.use_glove = kwargs.get('use_glove', 1)
        self.data = pandas.read_csv(data)
        self.glove_path = get_path('models/glove/glove.6B.300d.txt')
        self.nass_embedding_path = get_path('models/word2vec/nassai_word2vec.vec')
        self.encoding = "utf-8"
        self.word2vecmodel = None
        self.max_sequence_length = 1000
        self.max_num_words = 20000
        self.embedding_dim = 300
        self.validation_split = 0.2
        self.num_words = None
        self.embedding_matrix = None
        self.result = {"type": "word2vec"}

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
        x = Dense(256, name='Dense_1', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(128, name='Dense_2', activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(8, name='output', activation='softmax')(x)
        model = Model(sequence_input, predictions)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc', f1])
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
                      metrics=['acc', f1])
        return model

    def mlp_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(self.max_sequence_length,)))
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
            print("Loading word2vec embedding")
            model = Word2Vec.load(self.nass_embedding_path)
            embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

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
        all_words = set(w for words in texts for w in words)
        for word in all_words:
            index = word_index.get(word)
            if index > self.max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                    self.embedding_matrix[index] = embedding_vector

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

            y_pred = model.predict(test_data)

            numpy.savez_compressed('test_and_pred', test=test_data, predictions=y_pred)

            return evaluate_and_log(model, test_data, y_test, self.result)

        else:
            train_data, test_data, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            pipelist = [("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index, self.embedding_dim))]

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
            show_report(y_test, y_pred, self.data['bill_class'].unique(), self.result)

    def run_validation(self, clf, train, y_train):
        scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
        print("Running Validation on {0}".format(clf))
        print()
        cv = ShuffleSplit(n_splits=self.cv, test_size=0.2, random_state=42)
        accuracies = cross_validate(clf, train, y_train, scoring=scoring, cv=cv)
        print(accuracies)
        return True
