import pandas
import numpy
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Doc2Vec
from keras.models import Sequential
from keras.layers import Dropout, Dense
import keras_metrics

from code import get_path,VECTOR_SIZE, TEST_SIZE, VALIDATION_SIZE
from code.utils import handle_format


def get_vectors(model, data, get_for='train'):
    def vec(corpus_size, get_for):
            vec = numpy.zeros((corpus_size, VECTOR_SIZE))
            for i in range(0, corpus_size):
                prefix = get_for + '_' + str(i)
                vec[i] = model.docvecs[prefix]
            return vec
    vectors = handle_format(data, True)
    vectors = vec(len(vectors), get_for)
    return vectors


def encode_label(data):
    le = LabelEncoder()
    fit = le.fit_transform(data)
    output = np_utils.to_categorical(fit)
    return output


def build_model():
    model = Sequential()
    model.add(Dense(256, input_dim=VECTOR_SIZE, init='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, init='normal', activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', keras_metrics.f1_score()])
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
    print(train_vectors)
    print(y_train)
    keras_model = build_model()
    print(keras_model.summary())
    print()
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




