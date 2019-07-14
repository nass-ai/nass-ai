import numpy
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from code.utils import get_path, cache, MeanEmbeddingVectorizer


@cache
def get_glove():
    print("Loading Glove")
    embeddings_index = {}
    glove_path = get_path('models/glove/glove.6B.300d.txt')
    with open(glove_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = numpy.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
        return embeddings_index


@cache
def load_word2vec():
    print("Loading word2vec embedding")
    nass_embedding_path = get_path('models/word2vec/nassai_word2vec.vec')
    model = Word2Vec.load(nass_embedding_path)
    embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))
    return embeddings_index


def prep(x):
    return list(map(str, x))


class SklearnClassifierWrapper(object):
    def __init__(self, model, use_glove=True, name="SklearnClassifierWrapper"):
        """
        Classifier made up of a pipeline with a count vectorizer + given model
        :param model: a sklearn-like classifier (with fit, predict and predict_proba)
        :param tfidf: if True wil use TfidfVectorizer, otherwise CountVectorizer; defaults to False
        """
        self.embedding_index = {}
        if use_glove:
            self.embedding_index = get_glove()
        else:
            self.embedding_index = load_word2vec()

        self.clf = Pipeline([('mean_embedding_vectorizer', MeanEmbeddingVectorizer(self.embedding_index, 300)), ("model", model)])
        self.name = name

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def predict(self, X):
        return self.clf.predict(X)

    def __str__(self):
        return self.name


class MultNB(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, **kwargs):
        self.name = "MultinomialNB(ngram_n=%s)" % ngram_n
        super(MultNB, self).__init__(MultinomialNB(), name=self.name)


class BernNB(SklearnClassifierWrapper):
    def __init__(self, **kwargs):
        super(BernNB, self).__init__(BernoulliNB())


class SVM(SklearnClassifierWrapper):
    def __init__(self, kernel='linear', probability=False, **kwargs):
        super(SVM, self).__init__(SVC(kernel=kernel, C=10, gamma=0.0001, probability=probability))


class LinearSVM(SklearnClassifierWrapper):
    def __init__(self, kernel='linear', **kwargs):
        super(LinearSVM, self).__init__(LinearSVC(C=10))


class RandomForest(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, **kwargs):
        super(RandomForest, self).__init__(RandomForestClassifier())
