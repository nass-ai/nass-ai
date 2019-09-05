import pandas
from gensim.models import Word2Vec, Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from code.utils import handle_format, get_path


class BuildEmbeddingModel(object):
    def __init__(self, data, embedding_type, **kwargs):
        self.embedding_type = embedding_type
        self.data = pandas.read_csv(data)
        self.doc2vec_mode = kwargs.get('doc2vec_mode', 0)
        self.word2vec_mode = kwargs.get('word2vec_mode', 0)
        self.epoch = kwargs.get('epoch', 10)
        self.batch = kwargs.get('batch', 100)
        self.embedding_dim = 300

    def build_model(self):
        texts = self.data.clean_text
        if self.embedding_type == "word2vec":
            model_data = [word.split(' ') for word in texts]
            print("Initializing {0} model".format(self.embedding_type))
            model = Word2Vec(size=self.embedding_dim, window=5, min_count=5, workers=2, hs=1, sg=self.word2vec_mode, negative=5, alpha=0.065, min_alpha=0.065, max_vocab_size=5000)
        else:
            train, test = train_test_split(texts, random_state=42, test_size=0.2)
            print("Initializing {0} model".format(self.embedding_type))
            print("Tagging docs ...")
            train_formatted = handle_format(train)
            test_formatted = handle_format(test, False)
            print("Tagging Done ...")
            model_data = train_formatted + test_formatted
            model = Doc2Vec(dm=self.doc2vec_mode, vector_size=self.embedding_dim, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
        print("Building Model")
        model.build_vocab(model_data)
        return self.train_model(model, model_data)

    def train_model(self, model, model_data):
        print('Training Model')
        print("Training {0} {1}".format(len(model_data), "words" if self.embedding_type == "word2vec" else "doc2vec"))
        model.train(utils.shuffle([x for x in tqdm(model_data)]), total_examples=len(model_data), epochs=self.epoch)
        print("Training complete. Saving model")
        if self.embedding_type == "word2vec":
            model_path = get_path('models/word2vec') + '/nassai_word2vec.vec'.format("cbow" if self.doc2vec_mode else "skipgram")
        else:
            model_path = get_path('models/doc2vec') + '/nassai_{0}_doc2vec.vec'.format("dbow" if self.doc2vec_mode else "dm")
        model.save(model_path)
        return True
