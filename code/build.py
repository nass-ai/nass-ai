import pandas
from gensim.models import Word2Vec, Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from code.utils import handle_format, get_path


logging.getLogger().setLevel(logging.INFO)


class Embedding(object):
    def __init__(self, data, embedding_type, **kwargs):
        self.type = embedding_type
        self.data = pandas.read_csv(data)
        self.cbow = kwargs.get('cbow', 1)
        self.dbow = kwargs.get('dbow', 1)
        self.epoch = kwargs.get('epoch', 5)
        self.dim = kwargs.get('dim', 300)

    def build(self):
        texts = self.data.clean_text
        if self.type == "word2vec":
            model_data = [word.split(' ') for word in texts]
            logging.info("Initializing {0} model".format(self.type))
            model = Word2Vec(size=self.dim, window=5, min_count=5, workers=2, hs=1, sg=self.cbow, negative=5, alpha=0.065, min_alpha=0.065, max_vocab_size=2000)
        else:
            train, test = train_test_split(texts, random_state=42, test_size=0.2)
            logging.info("Initializing {0} model".format(self.type))
            logging.info("Tagging docs ...")
            train_formatted = handle_format(train)
            test_formatted = handle_format(test, False)
            logging.info("Tagging Done ...")
            model_data = train_formatted + test_formatted
            model = Doc2Vec(dm=self.dbow, vector_size=self.dim, negative=5, min_count=1, alpha=0.065, min_alpha=0.065, max_vocab_size=1500, dbow_words=1, verbose=1)
        logging.info("Building {0} Model".format(self.type))
        model.build_vocab(model_data)
        return self.train(model, model_data)

    def train(self, model, model_data):
        model.train(utils.shuffle([x for x in tqdm(model_data)]), total_examples=len(model_data), epochs=self.epoch)
        logging.info("Training complete. Saving model")
        model_path = get_path('models/word2vec') + '/nassai_word2vec.vec'.format(self.type) if self.type == "word2vec" else get_path('models/doc2vec') + '/nassai_doc2vec.vec'.format(
            self.type)
        model.save(model_path)
        return True


