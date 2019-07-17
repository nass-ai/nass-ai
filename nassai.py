import csv
import os
from datetime import datetime

import click

from code.custom import LSTMClassifier, BLSTM2DCNN, FCholletCNN, YKimCNN
from code.mlp import mlp_model
from code.sklearn_classifiers import BernNB, SVM, LinearSVM, MLP
from code.train import train
from code.utils import get_path, load_model
from code.build import BuildEmbeddingModel

from keras import backend as K
import tensorflow as tf

NUM_PARALLEL_EXEC_UNITS = 4
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                        allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

K.set_session(session)


@click.command()
@click.argument('action', type=click.Choice(['preprocess', 'build_embedding', 'train', 'predict']))
@click.option('--dbow', type=click.INT, default=1, help='Uses DBOW if true. DM if false.')
@click.option('--use_glove', type=click.INT, default=1, help='Train classifier with glove embedding or prebuilt embedding from this data.')
@click.option('--cbow', type=click.INT, default=1, help='Uses DBOW if true. DM if false.')
@click.option('--batch', type=click.INT, default=200, help='Batch for training keras model')
@click.option('--epoch', type=click.INT, default=200, help='Epoch for training keras model')
@click.option('--using', type=click.Choice(['sklearn', 'keras']), help='Algorithm to train data on.')
@click.option('--mode', type=click.Choice(['tfidf', 'doc2vec', 'word2vec']), help='Algorithm to train data on.')
@click.option('--text', type=click.STRING, help="String to predict for")
def nassai_cli(action, cbow, batch, epoch, using, dbow, mode, text, use_glove=1):
    base_data_path = get_path('data') + "/final_with_dates.csv"
    clean_data_path = get_path('data') + "/clean_data.csv"

    if action == "preprocess":
        from code import preprocessing
        return preprocessing.preprocess_data(base_data_path)
    elif action == "build_embedding":
        if mode == "doc2vec":
            builder = BuildEmbeddingModel(embedding_type="doc2vec", data=clean_data_path, doc2vec_mode=dbow, epoch=epoch, batch=batch)
            return builder.build_model()
        builder = BuildEmbeddingModel(embedding_type="word2vec", data=clean_data_path, word2vec_mode=cbow, epoch=epoch, batch=batch)
        return builder.build_model()
    elif action == "train":
        if mode == "doc2vec":
            embedding = get_path('models') + '/doc2vec/nassai_dbow_doc2vec.vec'
            if using == "sklearn":
                model_list = [("bnb_mean_embedding", BernNB(use_glove=False, embedding_path=embedding, use_tfidf=False, tfidf="mean_embedding")),
                              ("svm_mean_embedding", SVM(use_glove=False, use_tfidf=False, embedding_path=embedding, tfidf="mean_embedding")),
                              ("linear_svm_mean_embedding", LinearSVM(use_glove=False, use_tfidf=False, embedding_path=embedding, tfidf="mean_embedding")),

                              ("bnb_tfidfemmbedding", BernNB(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer")),
                              ("svm_tfidfembedding", (SVM(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"))),
                              ("linear_svm_tfidfembedding", LinearSVM(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"))

                              ]
            else:
                model_list = [('mlp', mlp_model), ("mlp_mean_embedding", MLP(use_glove=True, use_tfidf=False, tfidf="mean_embedding"), 1),
                              ("mlp_tfidfemmbedding", MLP(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"), 1)]
        elif mode == "word2vec":
            if using == "sklearn":
                model_list = [("bnb_mean_embedding", BernNB(use_glove=True, use_tfidf=False, tfidf="mean_embedding")),
                              ("svm_mean_embedding", (SVM(use_glove=True, use_tfidf=False, tfidf="mean_embedding"))),
                              ("linear_svm_mean_embedding", LinearSVM(use_glove=True, use_tfidf=False, tfidf="mean_embedding")),

                              ("bnb_tfidfemmbedding", BernNB(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer")),
                              ("svm_tfidfembedding", (SVM(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"))),
                              ("linear_svm_tfidfembedding", LinearSVM(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"))]
            else:
                model_list = [('mlp', mlp_model), ("mlp_mean_embedding", MLP(use_glove=True, use_tfidf=False, tfidf="mean_embedding"), 1),
                              ("mlp_tfidfemmbedding", MLP(use_glove=True, use_tfidf=False, tfidf="tfidf_embedding_vectorizer"), 1)]
        else:
            if using != "sklearn":
                model_list = [("bilstm-cnn", BLSTM2DCNN(train_embeddings=True, batch=True, use_glove=False, units=256)),
                              ("LSTMClassifier", LSTMClassifier(train_embeddings=True, batch=True, use_glove=False, units=256, layers=4))]
            else:
                model_list = [("bnb", BernNB(use_glove=False, use_tfidf=True)),
                              ("svm", (SVM(use_glove=False, use_tfidf=True))), ("linear_svm", LinearSVM(use_glove=False, use_tfidf=True))]

        return run(model_list, mode=mode, using=using, layers=4, dropout_rate=0.25)

    else:
        model = load_model(mode, using)
        pred = model.predict([text])
        click.echo("TEXT : {}".format(text))
        print()
        click.echo("PREDICTION: {}".format(pred))


def run(model_list, mode, **kwargs):
    records = {}
    results_path = get_path('data') + '/results.csv'
    clean_data_path = get_path('data') + '/clean_data.csv'
    print("TRAINING : {}".format(mode))
    for model in model_list:
        print("Current Model : {}".format(model))
        score = train(clf=model, data=clean_data_path, **kwargs)
        records.update({
            'date': datetime.now(),
            'f1': score,
            'mode': mode,
            'model_name': model[0],
            'using': kwargs.get('using')
        })
        with open(results_path, 'a') as f:
            w = csv.DictWriter(f, records.keys())
            w.writerow(records)


if __name__ == "__main__":
    nassai_cli()
