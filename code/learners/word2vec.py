import pandas
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from code.utils import get_path, show_report, cache

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_word2vec(clf, data, mode, **kwargs):
    clf = clf
    epoch_count = kwargs.get('epoch', 10)
    batch = kwargs.get('batch', 10)
    cv = kwargs.get('cv', 5)
    cbow = kwargs.get('cbow', 1)
    use_glove = kwargs.get('use_glove', 1)
    data = pandas.read_csv(data)
    glove_path = get_path('models/glove/glove.6B.300d.txt')
    nass_embedding_path = get_path('models/word2vec/nassai_word2vec.vec')
    encoding = "utf-8"
    word2vecmodel = None
    max_sequence_length = 1000
    max_num_words = 20000
    embedding_dim = 300
    validation_split = 0.2
    num_words = None
    embedding_matrix = None

    texts = data.apply(lambda r: simple_preprocess(r['clean_text'], min_len=2), axis=1)
    labels = data.bill_class
    if mode == "sklearn":
            train_data, test_data, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            pipelist = [clf]
            pipeline = Pipeline(pipelist)
            print(pipeline.steps)
            pipeline.fit(train_data, y_train)
            print("Scoring ...")
            y_pred = pipeline.predict(test_data)
            return show_report(y_test, y_pred, data['bill_class'].unique())

