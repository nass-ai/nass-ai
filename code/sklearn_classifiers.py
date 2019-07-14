from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC


def prep(x):
    return list(map(str, x))


class SklearnClassifierWrapper(object):
    def __init__(self, model, ngram_n=1, name="SklearnClassifierWrapper"):
        """
        Classifier made up of a pipeline with a count vectorizer + given model
        :param model: a sklearn-like classifier (with fit, predict and predict_proba)
        :param tfidf: if True wil use TfidfVectorizer, otherwise CountVectorizer; defaults to False
        """
        vectorizer_class = TfidfVectorizer
        vectorizer = vectorizer_class(
            preprocessor=lambda x: prep(x),
            tokenizer=lambda x: x,
            ngram_range=(1, 2))

        self.clf = Pipeline([('vectorizer', vectorizer), ('tfidf', TfidfTransformer(use_idf=True)), ('model', model)])
        print(self.clf.steps)
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
        super(MultNB, self).__init__(MultinomialNB(), ngram_n=ngram_n, name=self.name)


class BernNB(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, **kwargs):
        self.name = "BernoulliNB(ngram_n=%s)" % (ngram_n)
        super(BernNB, self).__init__(BernoulliNB(), ngram_n, self.name)


class SVM(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, kernel='linear', probability=False, **kwargs):
        super(SVM, self).__init__(SVC(kernel=kernel, C=10, gamma=0.0001, probability=probability), ngram_n)
        self.name = "SVC(ngram_n=%s, kernel=%s)" % (ngram_n, kernel)


class LinearSVM(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, kernel='linear', **kwargs):
        super(LinearSVM, self).__init__(LinearSVC(C=10), ngram_n)
        self.name = "SVC(ngram_n=%s, kernel=%s)" % (ngram_n, kernel)


class RandomForest(SklearnClassifierWrapper):
    def __init__(self, ngram_n=1, **kwargs):
        super(RandomForest, self).__init__(RandomForestClassifier(), ngram_n)
        self.name = "SVC(ngram_n=%s,)" % ngram_n
