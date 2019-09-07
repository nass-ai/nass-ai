import pandas
import keras
from datetime import datetime
from gensim.utils import simple_preprocess
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from code.sklearn_classifiers import SklearnClassifierWrapper, MLP
from code.utils import show_report, batch_generator, get_path, cache


def star(f):
    return lambda args: f(*args)


def prepare_data(prep=False, do_decode=False):
    data = pandas.read_csv(get_path('data') + "/clean_data.csv")
    if prep:
        text = data.apply(lambda r: simple_preprocess(r['clean_text'], min_len=3), axis=1)
    else:
        text = data.clean_text.values
    if not do_decode:
        labels = data.bill_class
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
    else:
        labels = data.bill_class
    tok = Tokenizer(num_words=30000)
    tok.fit_on_texts(text)
    word_counts = tok.word_counts
    vocab = [''] + [w for (w, _) in sorted(word_counts.items(), key=star(lambda _, c: -c))]
    vocab = [i for i in vocab if len(i) < 15 and len(i)> 1]
    return text, labels, data['bill_class'].unique(), vocab, tok


def train(clf, **kwargs):
    batch_size = 300
    model = clf[1]
    logdir = "logs/mlp/{0}_{1}".format(clf[0], datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    early_stopping = EarlyStopping(
                monitor='val_loss', patience=3)

    if not isinstance(model, SklearnClassifierWrapper):
        texts, labels, unique, vocab, tok = prepare_data(do_decode=False)
        print("Length of Vocab : {}".format(len(vocab)))
        train_data, test_data = train_test_split(texts, test_size=0.2, random_state=42)
        y_train, y_test = train_test_split(labels, test_size=0.2, random_state=42)

        if clf[0].startswith("mlp_"):
            pipe = Pipeline([("vectorizer", TfidfVectorizer(min_df=0.2)), ('tfidf', TfidfTransformer(use_idf=True)), ("model", KerasClassifier(build_fn=model, callbacks =[tensorboard_callback, early_stopping], epochs=50, batch_size=512))])
            return fit_and_report(pipe, train_data, test_data, y_train, y_test, unique, name=kwargs.get('name'))

        else:
            train_data = tok.texts_to_matrix(train_data)
            test_data = tok.texts_to_matrix(test_data)
            early_stopping = EarlyStopping(
                monitor='val_loss', patience=3)

            train_padded = pad_sequences(train_data, 1000)
            test_padded = pad_sequences(test_data, 1000)

            if kwargs.get('use_generator'):
                train_gen = batch_generator(train_padded, y_train, batch_size)
                valid_gen = batch_generator(test_padded, y_test, batch_size)

                model = model.fit_generator(
                    generator=train_gen,
                    epochs=5,
                    validation_data=valid_gen,
                    steps_per_epoch=valid_gen.shape[0] // batch_size,
                    validation_steps=test_padded.shape[0] // batch_size,
                    callbacks=[early_stopping, tensorboard_callback])
                return fit_and_report(model, train_data, test_data, y_train, y_test, unique, kwargs.get('name'))

            else:
                print("Length of Vocab : {}".format(len(vocab)))
                model = model.set_up(vocab)
                model = model.fit(
                    train_padded, y_train,
                    validation_data=[test_padded, y_test]
                )
                return fit_and_report(model, train_data, test_data, y_train, y_test, unique, kwargs.get('name'))

    if isinstance(model, MLP):
        texts, labels, unique, vocab, tok = prepare_data(do_decode=True)
        train_data, test_data = train_test_split(texts, test_size=0.2, random_state=42)
        y_train, y_test = train_test_split(labels, test_size=0.2, random_state=42)
        pipe = Pipeline([("vectorizer", TfidfVectorizer(min_df=0.2)), ('tfidf', TfidfTransformer(use_idf=True)), ("model", KerasClassifier(build_fn=model, callbacks =[tensorboard_callback, early_stopping], layers=kwargs.get('layers'),
                                                                                                                                           dropout_rate=kwargs.get('dropout_rate'), epochs=50, batch_size=512))])
        return fit_and_report(pipe, train_data, test_data, y_train, y_test, unique, kwargs.get('name'))
        

    else:
        texts, labels, unique, vocab, tok = prepare_data(prep=True)
        print("Length of Vocab : {}".format(len(vocab)))
        train_data, test_data = train_test_split(texts, test_size=0.2, random_state=42)
        y_train, y_test = train_test_split(labels, test_size=0.2, random_state=42)
        model.set_up(vocab)
        return fit_and_report(model, train_data, test_data, y_train, y_test, unique, name=kwargs.get('name'))


def fit_and_report(model, train_data, test_data, y_train, y_test, labels, name):
    model.fit(train_data, y_train)
    y_pred = model.predict(test_data)

    import numpy
    numpy.savez_compressed("{}.npz".format(name), test=y_test, predictions=y_pred)

    return show_report(y_test, y_pred, labels)
