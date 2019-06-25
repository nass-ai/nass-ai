from time import time

import pandas
from gensim.models import Doc2Vec
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from code import get_path
from code.learners import get_vectors


class NassAIDoc2Vec:
    def __init__(self, clf, data, **kwargs):
        self.clf = clf
        self.dbow = kwargs.get('dbow', True)
        self.epoch_count = kwargs.get('epoch', 10)
        self.data = pandas.read_csv(data)

    def train(self):
        model_path = get_path('models/doc2vec')
        model_path = model_path + '/dbow_doc2vec.vec' if self.dbow else model_path + '/dm_doc2vec.vec'
        print("Loading model")
        doc2vec_model = Doc2Vec.load(model_path)
        print("Model Loaded")

        self.data['category_id'] = self.data['bill_class'].factorize()[0]
        train_features, test_features, y_train, y_test = train_test_split(self.data.clean_text, self.data.category_id, test_size=0.2, random_state=42)
        train_features, val_features, y_train, y_val = train_test_split(train_features, y_train, test_size=0.2, random_state=42)

        train_vectors = get_vectors(doc2vec_model, train_features)
        test_vectors = get_vectors(doc2vec_model, test_features, 'test')
        val_vectors = get_vectors(doc2vec_model, val_features)

        clf_map = {"mlp": MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(256,), activation='relu'), "mnb": MultinomialNB(), "svm": SVC()}
        if self.clf in ['mnb', 'svm', 'mlp']:
            clf = clf_map.get(self.clf)
            print("Fitting data...")
            t0 = time()
            clf.fit(train_vectors, y_train)
            print("done in %0.3fs" % (time() - t0))
            print()
            print("Running Validation on {0}".format(clf))
            scoring = {'acc': 'accuracy', 'f1_micro': 'f1_micro'}
            for epoch in range(self.epoch_count):
                accuracies = cross_validate(clf, val_vectors, y_val, scoring=scoring, cv=5)
                for fold, accuracy in enumerate(accuracies):
                    print("{0}: \nAccuracy: {1} \nF1:{2}".format(fold, accuracies['test_acc'][fold - 1], accuracies['test_f1_micro'][fold - 1]))

            print("Scoring ...")
            y_pred = clf.predict(test_vectors)
            from sklearn import metrics

            print(metrics.classification_report(y_test, y_pred, target_names=self.data['bill_class'].unique()))
            return True
