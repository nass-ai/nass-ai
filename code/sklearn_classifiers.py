import numpy
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from code.utils import get_path, cache, MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer, encode_label, f1


@cache
def get_glove(vocab):
    print("Loading Glove")
    embeddings_matrix = {}
    glove_path = get_path('models/glove/glove.6B.300d.txt')
    with open(glove_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            if word in vocab:
                coefs = numpy.fromstring(coefs, 'f', sep=' ')
                embeddings_matrix[word] = coefs
        return embeddings_matrix


@cache
def load_word2vec(embedding_path):
    print("Loading embedding")
    model = Word2Vec.load(embedding_path)
    embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))
    return embeddings_index


def prep(x):
    return list(map(str, x))


class SklearnClassifierWrapper(object):
    def __init__(self, model, use_glove, use_tfidf, tfidf=None, embedding_path=None, name="SklearnClassifierWrapper"):
        """
        Classifier made up of a pipeline with a count vectorizer + given model
        :param model: a sklearn-like classifier (with fit, predict and predict_proba)
        :param tfidf: if True wil use TfidfVectorizer, otherwise CountVectorizer; defaults to False
        """
        self.embedding_index = {}
        self.embedding_path = embedding_path
        self.use_glove = use_glove
        self.vocab = None
        self.use_tfidf = use_tfidf
        self.tfidf = tfidf
        self.name = name
        self.model = model
        self.clf = None

    def set_up(self, vocab):
        self.vocab = vocab
        if not self.use_glove and not self.use_tfidf:
            self.embedding_index = load_word2vec(self.embedding_path)

        elif self.use_glove:
            self.embedding_index = get_glove(vocab=self.vocab)

        if self.use_tfidf:
            vectorizer_class = HashingVectorizer
            vectorizer = vectorizer_class(
                preprocessor=lambda x: map(str, x),
                tokenizer=lambda x: x)
            self.clf = Pipeline([('mean_embedding_vectorizer', vectorizer), ("transformer", TfidfTransformer()), ("model", self.model)])
        else:
            if self.tfidf == "mean_embeding":
                self.clf = Pipeline([('mean_embedding_vectorizer', MeanEmbeddingVectorizer(self.embedding_index, 300)), ("model", self.model)])
            else:
                self.clf = Pipeline([('tfidf_embedding_vectorizer', TfidfEmbeddingVectorizer(self.embedding_index, 300)), ("model", self.model)])

        return self

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return self.name


class BernNB(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(BernNB, self).__init__(BernoulliNB(), **kwargs)


class SVM(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(SVM, self).__init__(SVC(C=10, gamma=0.0001, cache_size=512, class_weight=None, coef0=0.0,
                                      decision_function_shape='ovr', degree=1, kernel='linear',
                                      max_iter=13206119.0, probability=False, random_state=0, shrinking=False,
                                      tol=0.0023491958481415094, verbose=False), **kwargs)


class LinearSVM(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(LinearSVM, self).__init__(LinearSVC(C=10), **kwargs)


class RandomForest(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(RandomForest, self).__init__(RandomForestClassifier(), **kwargs)


class MLPCLF(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(MLPCLF, self).__init__(MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(512, 256, 128), activation='relu'), **kwargs)


class MLP(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        self.model = self.mlp_model()
        super(MLP, self).__init__(self.model, **kwargs)

    def mlp_model(self,
                  layers=1,
                  units=256,
                  dropout_rate=0.5):

        max_seq_len = 300

        model = Sequential()
        for i in range(layers):
            if i == 0:
                model.add(Dense(units, input_shape=(max_seq_len,)))
            else:
                model.add(Dense(units))
            model.add(Activation('relu'))
            model.add(Dropout(dropout_rate))
        model.add(Dense(8, name="output_dense"))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy', f1])
        return model
