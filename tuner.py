from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import train_test_split

from code.custom import YKimCNN, BLSTM2DCNN, LSTMClassifier
from code.mlp import MLP
from code.train import prepare_data
from code.utils import cache
import pprint

import numpy as np


# from benchmarks import benchmark_with_early_stopping, cache

@cache
def benchmark_with_early_stopping(model_class, model_params=None):
    """same as benchmark but fits with validation data to allow the model to do early stopping
    Works with all models from keras_models
    :param model_class: class of the model to instantiate, must have fit(X, y, validation_data)
        method and 'history' attribute
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :return: best_loss, best_score, best_epoch
    """
    if model_params is None:
        model_params = {}

    X, y, _, vocab, tok = prepare_data(prep=True, do_decode=False)
    model_params['vocab'] = vocab
    model_params['tok'] = tok
    model = model_class(**model_params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model.fit(X_train, y_train, validation_data=[X_test, y_test])
    best_loss = np.min(model.history.history['val_loss'])
    best_acc = np.max(model.history.history['val_acc'])
    best_epoch = np.argmin(model.history.history['val_loss']) + 1

    print(model, "acc", best_acc, "loss",  best_loss, "epochs", best_epoch)
    return best_loss, best_acc, best_epoch


def fix_ints(d):
    return {
        k: int(v) if (isinstance(v, float) and int(v) == v) else v
        for k, v in d.items()
    }


@cache
def hyperopt_me_like_one_of_your_french_girls(
        classifier, space, max_evals):
    def objective(args):
        best_loss, best_acc, best_epoch = benchmark_with_early_stopping(classifier,
                                                                        fix_ints(args))
        return {
            'loss': best_loss,
            'accuracy': best_acc,
            'epochs': best_epoch,
            'status': STATUS_OK
        }

    trials = Trials()
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials)

    return trials


if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)

    trials = hyperopt_me_like_one_of_your_french_girls(
        YKimCNN, {
            'units': hp.quniform('units', 10, 100, 10),
            'dropout_rates': [hp.uniform('dropout_1', 0.1, 0.9), hp.uniform('dropout_2', 0.1, 0.9)],
            'num_filters': hp.quniform('num_filters', 5, 100, 5),
            'filter_sizes': hp.choice('filter_sizes', [
                [3, 8],
                [3, 5],
                [3, 6],
                [3, 4, 5],
                [3, 5, 8],
                [3, 5, 7],
                [3, 4, 5, 6]
            ]),
            # 'embedding_dim': hp.quniform('embedding_dim', 5, 60, 5),
            # 'epochs': 200,
            # 'max_seq_len': 50
        }, max_evals=100)

    print('\n\nYKimCNN trainable embedding')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        YKimCNN, {
            'units': hp.quniform('units', 10, 100, 10),
            'dropout_rates': [hp.uniform('dropout_1', 0.1, 0.9), hp.uniform('dropout_2', 0.1, 0.9)],
            'num_filters': hp.quniform('num_filters', 5, 100, 5),
            'filter_sizes': hp.choice('filter_sizes', [
                [3, 8],
                [3, 5],
                [3, 6],
                [3, 4, 5],
                [3, 5, 8],
                [3, 5, 7],
                [3, 4, 5, 6]
            ]),
            # 'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            # 'epochs': 200,
            # 'max_seq_len': 50
        }, max_evals=100)

    print('\n\nYKimCNN glove embedding')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        BLSTM2DCNN, {
            'units': hp.quniform('units', 8, 128, 4),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            # 'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print('\n\nBLSTM with pretrained embedding')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        BLSTM2DCNN, {
            'units': hp.quniform('units', 8, 256, 1),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.9),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.9),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=100)

    print('\n\nBLSTM2DCNN with trainable embedding')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        MLP, {
            'layers': hp.quniform('layers', 1, 5, 1),
            'units': hp.quniform('units', 8, 2048, 1),
            'dropout_rate': hp.uniform('dropout_rate', 0.01, 0.99),
            # 'epochs': 200,
            'max_vocab_size': hp.quniform('max_vocab_size', 4000, 25000, 1000)
        }, max_evals=200)

    print('\n\nMLP')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, {
            'layers': hp.quniform('layers', 1, 4, 1),
            'units': hp.quniform('units', 8, 256, 1),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.9),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.9),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1)
        }, max_evals=100)

    print('\n\nLSTM')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 1),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embedding_dim': hp.quniform('embedding_dim', 2, 40, 1),
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print('\n\nBLSTM')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 4),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'bidirectional': True,
            'batch_size': 64
        }, max_evals=100)

    print('\n\nBLSTM with pretrained embedding')
    pp.pprint(trials.best_trial)

    trials = hyperopt_me_like_one_of_your_french_girls(
        LSTMClassifier, {
            'layers': hp.quniform('layers', 1, 3, 1),
            'units': hp.quniform('units', 8, 128, 4),
            # 'max_seq_len': 50,
            'dropout_rate': hp.uniform('dropout_rate', 0., 0.95),
            'rec_dropout_rate': hp.uniform('rec_dropout_rate', 0., 0.95),
            # 'epochs': 60,
            'optimizer': hp.choice('optimizer', ['adam', 'rmsprop']),
            'embeddings_path': '../data/glove.6B/glove.6B.100d.txt',
            'batch_size': 64
        }, max_evals=100)

    print('\n\nLSTM with pretrained embedding')
    pp.pprint(trials.best_trial)
