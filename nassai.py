import click
from code import get_path


@click.command()
@click.argument('action', type=click.STRING)
@click.option('--dbow_epoch', default=10, help='Number of epochs to train the dbow model for')
@click.option('--dbow_run_repeat', default=100, help='Number of times to repeat the epochs for. Ideally would train for be dbow_epoch * dbow_run_repeat')
@click.option('--dbow', type=click.BOOL, help='Uses DBOW if true. DM if false.')
@click.option('--keras_batch', type=click.INT, default=200, help='Batch for training keras model')
@click.option('--keras_epoch', type=click.INT, default=500, help='Epoch for training keras model')
def nassai_cli(action, dbow_run_repeat, dbow_epoch, dbow, keras_batch, keras_epoch):
    if action == "preprocess":
        from code import preprocessing
        data_path = get_path('data') + "/final_with_dates.csv"
        return preprocessing.preprocess_data(data_path)
    elif action == "train_doc2vec":
        from code.doc2vec_model import build_doc2vec_model, train_doc2vec_model
        data_path = get_path('data') + "/clean_data.csv"
        if not dbow:
            print("No Doc2Vec type passed. Defaulting to 'dbow'")
            model, formatted_data = build_doc2vec_model(data_path)
        else:
            model, formatted_data = build_doc2vec_model(data_path, dbow=dbow)
        return train_doc2vec_model(model, formatted_data, run_repeat=dbow_run_repeat, epochs=dbow_epoch, dbow=dbow)
    elif action == "train_doc2vec_mlp":
        from code.learners.doc2vec_mlp import train
        data_path = get_path('data') + "/clean_data.csv"
        return train(data_path, dbow=dbow, epochs=keras_epoch, batch=keras_batch)
    elif action == "train_doc2vec_bilstm":
        from code.learners.doc2vec_bilstm import train
        data_path = get_path('data') + "/clean_data.csv"
        return train(data_path, dbow=dbow, epochs=keras_epoch, batch=keras_batch)


if __name__ == "__main__":
    nassai_cli()
