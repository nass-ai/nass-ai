import numpy
import pandas
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.pipeline import Pipeline
from code.utils import get_path, show_report, cache
from code.utils import MeanEmbeddingVectorizer


@cache
def run_validation(clf, train, y_train):
    print("Running Validation on {0}".format(clf))
    print()
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    accuracies = cross_validate(clf, train, y_train, scoring='f1_macro', cv=cv)
    print(accuracies)
    return True


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

    @cache
    def train():
        print('Indexing word vectors.')

        embeddings_index = {}

        texts = data.apply(lambda r: simple_preprocess(r['clean_text'], min_len=2), axis=1)
        print('Found %s texts.' % len(texts))
        labels = data.bill_class

        if use_glove:
            with open(glove_path) as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = numpy.fromstring(coefs, 'f', sep=' ')
                    embeddings_index[word] = coefs

        else:
            print("Loading word2vec embedding")
            model = Word2Vec.load(nass_embedding_path)
            embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

        print('Found %s word vectors.' % len(embeddings_index))

        if mode == "sklearn":
            train_data, test_data, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
            pipelist = [("word2vec vectorizer", MeanEmbeddingVectorizer(embeddings_index, embedding_dim)), clf]
            pipeline = Pipeline(pipelist)
            pipeline.fit(train_data, test_data)
            print("Scoring ...")
            run_validation(pipeline, train_data, y_train)
            y_pred = pipeline.predict(test_data)
            return show_report(y_test, y_pred, data['bill_class'].unique())

    return train()

