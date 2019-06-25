import click
from code import get_path
from code.learners.doc2vec import NassAIDoc2Vec
from code.learners.tfid_models import NassAITfidf


@click.command()
@click.argument('action', type=click.STRING)
@click.option('--dbow_epoch', default=10, help='Number of epochs to train the dbow model for')
@click.option('--dbow_run_repeat', default=100, help='Number of times to repeat the epochs for. Ideally would train for be dbow_epoch * dbow_run_repeat')
@click.option('--dbow', type=click.BOOL, help='Uses DBOW if true. DM if false.')
@click.option('--batch', type=click.INT, default=200, help='Batch for training keras model')
@click.option('--epoch', type=click.INT, default=500, help='Epoch for training keras model')
@click.option('--clf', type=click.Choice(['cnn', 'bilstm', 'mnb', 'svm', 'mlp']), help='Algorithm to train data on.')
def nassai_cli(action, dbow_run_repeat, dbow_epoch, dbow, batch, epoch, clf="bilstm"):
    base_data_path = get_path('data') + "/final_with_dates.csv"
    clean_data_path = get_path('data') + "/clean_data.csv"
    if action == "preprocess":
        from code import preprocessing
        return preprocessing.preprocess_data(base_data_path)
    elif action == "build_doc2vec":
        from code.doc2vec_model import build_doc2vec_model, train_doc2vec_model
        if not dbow:
            print("No Doc2Vec type passed. Defaulting to 'dbow'")
            model, formatted_data = build_doc2vec_model(clean_data_path)
        else:
            model, formatted_data = build_doc2vec_model(clean_data_path, dbow=dbow)
        return train_doc2vec_model(model, formatted_data, run_repeat=dbow_run_repeat, epochs=dbow_epoch, dbow=dbow)
    elif action == "train_doc2vec":
        doc2vec = NassAIDoc2Vec(clf=clf, data=clean_data_path, dbow=dbow, epoch=epoch, batch=batch)
        return doc2vec.train()
    elif action == "train_tfidf":
        tf = NassAITfidf(clf=clf, data=clean_data_path, epoch=epoch)
        return tf.train()


if __name__ == "__main__":
    nassai_cli()
