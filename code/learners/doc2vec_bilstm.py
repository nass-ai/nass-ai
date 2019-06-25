import pandas
import numpy
from sklearn.model_selection import train_test_split
from gensim.models import Doc2Vec
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout, Dense, Bidirectional, LSTM
import keras_metrics

from code import get_path, VECTOR_SIZE, TEST_SIZE, VALIDATION_SIZE
from code.learners import get_vectors, encode_label


def build_model():
    print('Building LSTM model...')
    model = Sequential()
    model.add(Bidirectional(LSTM(256, activation="relu"), input_shape=(1, VECTOR_SIZE)))
    model.add(Dropout(0.3))
    model.add(Dense(8, init='normal', activation="softmax"))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', keras_metrics.f1_score()])
    print('LSTM model built.')
    return model


def train(data, epochs=500, batch=200, dbow=True):
    model_path = get_path('models/doc2vec')
    model_path = model_path + '/dbow_doc2vec.vec' if dbow else model_path + '/dm_doc2vec.vec'
    print("Loading model")
    doc2vec_model = Doc2Vec.load(model_path)
    print("Model Loaded")
    data = pandas.read_csv(data)
    x_train, x_test, y_train, y_test = train_test_split(data.clean_text, data.bill_class, random_state=0, test_size=TEST_SIZE, stratify=data.bill_class)
    print("Splitting Data")
    train_vectors = get_vectors(doc2vec_model, x_train)
    test_vectors = get_vectors(doc2vec_model, x_test, 'test')
    y_train = encode_label(y_train)
    y_test = encode_label(y_test)
    keras_model = build_model()
    print(keras_model.summary())
    print()
    train_vectors = numpy.reshape(train_vectors, (train_vectors.shape[0], 1, train_vectors.shape[1]))
    print(train_vectors.shape)
    test_vectors = numpy.reshape(test_vectors, (test_vectors.shape[0], 1, test_vectors.shape[1]))
    fitted_model = keras_model.fit(train_vectors, y_train, validation_split=VALIDATION_SIZE, epochs=epochs, batch_size=batch)
    print('Model trained ...')
    print("Testing and printing results")
    print()
    print("Training Accuracy: %.2f%% \nValidation Accuracy: %.2f%%" % (100 * fitted_model.history['acc'][-1], 100 * fitted_model.history['val_acc'][-1]))
    print()
    scores = keras_model.evaluate(test_vectors, y_test, batch)
    print("Test Accuracy: %.2f%%" % (100 * scores[1]))
    print("Test F1: %.2f%%" % (100 * scores[2]))
    return True

