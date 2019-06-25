import pandas
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn import utils
from gensim.models import Doc2Vec
from code import get_path
from code.utils import handle_format

tqdm.pandas(desc="progress-bar")

VECTOR_SIZE = 300
TEST_SIZE = 0.2


def build_doc2vec_model(data, dbow=True):
    print('Initializing {model} Model'.format(model="DBOW" if dbow else "DM"))
    if dbow == "dbow":
        model = Doc2Vec(dm=0, vector_size=VECTOR_SIZE, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    else:
        model = Doc2Vec(dm=1, dm_mean=1, vector_size=VECTOR_SIZE, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    data = pandas.read_csv(data)
    train, test = train_test_split(data.clean_text, random_state=0, test_size=TEST_SIZE)
    train_formatted = handle_format(train)
    test_formatted = handle_format(test, False)
    final = train_formatted + test_formatted
    print('Building Vocabulary')
    model.build_vocab([x for x in tqdm(final)])
    print('Vocabulary Built')
    return model, final


def train_doc2vec_model(model, data, epochs=10, run_repeat=100, dbow=False):
    print('Training {model} Model'.format(model="DBOW" if dbow else "DM"))
    for epoch in range(run_repeat):
        print("Epoch : {}".format(epoch))
        model.train(utils.shuffle([x for x in tqdm(data)]), total_examples=len(data), epochs=epochs)
        model.alpha -= 0.002
        model.min_alpha = model.alpha
    print("Training complete. Saving model")
    model_path = get_path('models/doc2vec')
    model_path = model_path + '/dbow_doc2vec.vec' if dbow else model_path + '/dm_doc2vec.vec'
    model.save(model_path)
    return True
