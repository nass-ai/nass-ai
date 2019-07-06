import click
from code.utils import get_path, load_model
from code.learners.doc2vec import NassAIDoc2Vec
from code.learners.tfidf import NassAITfidf
from code.learners.word2vec import NassAIWord2Vec
from code.build import BuildEmbeddingModel


@click.command()
@click.argument('action', type=click.Choice(['preprocess', 'build_embedding', 'train', 'predict']))
@click.option('--dbow', type=click.INT, default=1, help='Uses DBOW if true. DM if false.')
@click.option('--use_glove', type=click.INT, default=1, help='Train classifier with glove embedding or prebuilt embedding from this data.')
@click.option('--cbow', type=click.INT, default=1, help='Uses DBOW if true. DM if false.')
@click.option('--batch', type=click.INT, default=200, help='Batch for training keras model')
@click.option('--epoch', type=click.INT, default=200, help='Epoch for training keras model')
@click.option('--clf', type=click.Choice(['cnn', 'bilstm', 'svm', 'mlp_sklearn', 'mlp', 'random_forest', 'logreg', 'mnb', 'best', 'svm_linear']), help='Algorithm to train data on.')
@click.option('--mode', type=click.Choice(['tfidf', 'doc2vec', 'word2vec']), help='Algorithm to train data on.')
@click.option('--text', type=click.STRING, help="String to predict for")
def nassai_cli(action, cbow, batch, epoch, clf, dbow, mode, text, use_glove=1):
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
            doc2vec = NassAIDoc2Vec(clf=clf, data=clean_data_path, use_glove=use_glove, dbow=dbow, epoch=epoch, batch=batch)
            return doc2vec.train()
        elif mode == "word2vec":
            word2vec = NassAIWord2Vec(clf=clf, data=clean_data_path, use_glove=use_glove, cbow=cbow, epoch=epoch, batch=batch)
            return word2vec.train()
        else:
            tfidf = NassAITfidf(clf=clf, data=clean_data_path, epoch=epoch)
            return tfidf.train()
    else:
        model = load_model(mode, clf)
        pred = model.predict([text])
        click.echo("TEXT : {}".format(text))
        print()
        click.echo("PREDICTION: {}".format(pred))


if __name__ == "__main__":
    nassai_cli()
