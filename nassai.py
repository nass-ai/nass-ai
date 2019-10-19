import csv
from datetime import datetime

import click

from code.custom import LSTMClassifier, BLSTM2DCNN, FCholletCNN, YKimCNN
from code.sklearn_classifiers import BernNB, SVM, LinearSVM, MLP
from code.train import train
from code.utils import get_path, load_model
from code.build import Embedding


@click.command()
@click.argument('action', type=click.Choice(['preprocess', 'build_embedding', 'train', 'predict']))
@click.option('--dbow/--dm', default=False, help="DM or DBOW for doc2vec. Defaults to DM.")
@click.option('--glove/--no-glove', default=True, help='Train classifier with glove embedding or prebuilt embedding from this data. Defaults to glove.')
@click.option('--cbow/--skipgram', default=True, help='Skipgram or cbow for word2vec. Defaults to CBOW')
@click.option('--batch', type=click.INT, default=100, help='Batch for trainings. Default is 100.')
@click.option('--epoch', type=click.INT, default=5, help='Epoch for training model. Default is 5.')
@click.option('--mode', type=click.Choice(['doc2vec', 'word2vec', 'all']), help='Algorithm to train data on.')
@click.option('--text', type=click.STRING, help="String to predict for")
@click.option('--data', type=click.Path(exists=True), help="NASS Data Crawl path")
def nassai_cli(action, batch, epoch, mode, text, data, cbow=True, dbow=False, glove=True):
    base_data_path = data
    clean_data_path = get_path('data') + "/clean_data.csv"

    if action == "preprocess":
        from code import preprocessing
        return preprocessing.preprocess_data(base_data_path)
    elif action == "build_embedding":
        if dbow:
            builder = Embedding(embedding_type="doc2vec", data=clean_data_path, dbow=dbow, epoch=epoch)
            return builder.build()
        builder = Embedding(embedding_type="word2vec", data=clean_data_path, cbow=cbow, epoch=epoch)
        return builder.build()
    elif action == "train":
        word2vec_embedding = get_path('models') + '/doc2vec/nassai_word2vec.vec'
        doc2vec_embedding = get_path('models') + '/doc2vec/nassai_doc2vec.vec'
        if mode == "doc2vec":
            model_list = [
                ("doc2vec_bnb_mean_embedding", BernNB(glove=glove, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),
                ("doc2vec_svm_mean_embedding", SVM(glove=glove, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),
                ("doc2vec_linear_svm_mean_embedding", LinearSVM(glove=False, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),

                ("doc2vec_bnb_tfidfemmbedding", BernNB(glove=glove, use_tfidf=True)),
                ("doc2vec_svm_tfidfembedding", (SVM(glove=glove, use_tfidf=True))),
                ("doc2vec_linear_svm_tfidfembedding", LinearSVM(glove=glove, use_tfidf=True)),

                ("lstm_doc2vec_glove", LSTMClassifier(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding, layers=4)),
                ("fchollet_cnn_doc2vec_glove", FCholletCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("bilstm_doc2vec_glove", BLSTM2DCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("ykimcnn_doc2vec_glove", YKimCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding))

            ]

        elif mode == "word2vec":
            model_list = [
                ("bnb_mean_embedding", BernNB(glove=glove, tfidf="mean_embedding")),
                ("svm_mean_embedding", (SVM(glove=glove, tfidf="mean_embedding"))),
                ("linear_svm_mean_embedding", LinearSVM(glove=glove, tfidf="mean_embedding")),

                ("bnb_tfidfemmbedding", BernNB(glove=glove, use_tfidf=True)),
                ("svm_tfidfembedding", (SVM(glove=glove, use_tfidf=True))),
                ("linear_svm_tfidfembedding", LinearSVM(glove=glove, use_tfidf=True)),
                # ("mlp_mean_embedding", MLP(glove=glove, tfidf="mean_embedding"), 1),
                # ("mlp_tfidfemmbedding", MLP(glove=glove, tfidf="tfidf_embedding_vectorizer"), 1)
            ]
        else:
            model_list = [
                ("word2vec_bnb_mean_embedding", BernNB(glove=glove, embedding_path=word2vec_embedding, tfidf="mean_embedding")),
                ("word2vec_svm_mean_embedding", SVM(glove=glove, embedding_path=word2vec_embedding, tfidf="mean_embedding")),
                ("word2vec_linear_svm_mean_embedding", LinearSVM(glove=False, embedding_path=word2vec_embedding, tfidf="mean_embedding")),
                ("lstm_word2vec_glove", LSTMClassifier(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding, layers=4)),
                ("fchollet_cnn_doc2vec_glove", FCholletCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("bilstm_word2vec_glove", BLSTM2DCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("ykimcnn_word2vec_glove", YKimCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),

                ("doc2vec_bnb_mean_embedding", BernNB(glove=glove, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),
                ("doc2vec_svm_mean_embedding", SVM(glove=glove, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),
                ("doc2vec_linear_svm_mean_embedding", LinearSVM(glove=glove, embedding_path=doc2vec_embedding, tfidf="mean_embedding")),
                ("lstm_doc2vec_glove", LSTMClassifier(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding, layers=4)),
                ("fchollet_cnn_doc2vec_glove", FCholletCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("bilstm_doc2vec_glove", BLSTM2DCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),
                ("ykimcnn_doc2vec_glove", YKimCNN(train_embeddings=False, batch=True, glove=glove, units=256, embedding_path=doc2vec_embedding)),

                ("doc2vec_bnb_tfidfemmbedding", BernNB(glove=glove, use_tfidf=True)),
                ("doc2vec_svm_tfidfembedding", (SVM(glove=glove, use_tfidf=True))),
                ("doc2vec_linear_svm_tfidfembedding", LinearSVM(glove=True, use_tfidf=True))

            ]
        return run(model_list, mode=mode, batch=batch, layers=4, dropout_rate=0.25)

    else:
        model = load_model(mode, '')
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
        score, duration = train(clf=model, data=clean_data_path, name="{}_{}".format(model[0], mode), **kwargs)
        records.update({
            'date': datetime.now(),
            'f1': score,
            'mode': mode,
            'duration': duration,
            'model_name': model[0],
        })
        print("{0} took {1} seconds.".format(model, duration))
        with open(results_path, 'a') as f:
            w = csv.DictWriter(f, records.keys())
            w.writerow(records)


if __name__ == "__main__":
    nassai_cli()
