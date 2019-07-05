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
            model = Word2Vec(size=self.embedding_dim, window=5, min_count=5, workers=2, hs=1, sg=self.word2vec_mode, negative=5, alpha=0.065, min_alpha=0.065)
        else:
            train, test = train_test_split(texts, random_state=42, test_size=0.2)
            print("Tagging docs ...")
            train_formatted = handle_format(train)
            test_formatted = handle_format(test, False)
            print("Tagging Done ...")
            model_data = train_formatted + test_formatted
            print("Initializing {0} model".format(self.embedding_type))
            model = Doc2Vec(dm=self.doc2vec_mode, vector_size=self.embedding_dim, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
        print("Building Model")
        model.build_vocab(model_data)
        return self.train_model(model, model_data)

    def train_model(self, model, model_data):
        print('Training Model')
        print("Training {0} {1}".format(len(model_data), "words" if self.embedding_type == "word2vec" else "doc2vec"))
        for epoch in range(5):
            print("Epoch : {}".format(epoch))
            model.train(utils.shuffle([x for x in tqdm(model_data)]), total_examples=len(model_data), epochs=self.epoch)
            model.alpha -= 0.002
            model.min_alpha = model.alpha
        print("Training complete. Saving model")
        model_path = get_path('models/doc2vec')
        model_path = model_path + '/nassai_word2vec.vec' if self.embedding_type == "word2vec" else model_path + '/nassai_doc2vec.vec'
        model.save(model_path)
        return True
