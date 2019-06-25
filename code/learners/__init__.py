from code.utils import handle_format
from sklearn.preprocessing import LabelEncoder
import numpy
from keras.utils import np_utils
from code import get_path,VECTOR_SIZE, TEST_SIZE, VALIDATION_SIZE


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
