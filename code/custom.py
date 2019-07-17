from keras.layers import Conv2D, MaxPool2D, \
    Reshape

from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Concatenate

from code.keras_classifiers import KerasTextClassifier


class BLSTM2DCNN(KerasTextClassifier):
    """based on https://arxiv.org/abs/1611.06639v1"""

    def __init__(
            self,
            batch,
            use_glove,
            train_embeddings=False,
            embedding_path=None,
            units=128,
            dropout_rates=0.2,
            rec_dropout_rate=0.2,
            conv_filters=32,
    ):
        super(BLSTM2DCNN, self).__init__(
            batch=batch,
            use_glove=use_glove,
            train_embeddings=train_embeddings,
            embedding_path=embedding_path
        )

        self.params['units'] = units
        self.params['dropout_rates'] = dropout_rates
        self.params['rec_dropout_rate'] = rec_dropout_rate
        self.params['conv_filters'] = conv_filters

    def transform_embedded_sequences(self, embedded_sequences):
        x = Dropout(self.params['dropout_rates'])(embedded_sequences)
        x = Bidirectional(LSTM(
            self.params['units'],
            dropout=self.params['dropout_rates'],
            recurrent_dropout=self.params['rec_dropout_rate'],
            return_sequences=True))(x)
        x = Reshape((2 * self.max_seq_len, self.params['units'], 1))(x)
        x = Conv2D(self.params['conv_filters'], (3, 3))(x)
        x = MaxPool2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds


class FCholletCNN(KerasTextClassifier):
    """Based on
    https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
    except with trainable embeddings instead of pretrained from GloVe"""

    def __init__(
            self,
            batch,
            use_glove,
            train_embeddings=False,
            embedding_path=None,
            units=128,
            dropout_rates=0.25):
        super(FCholletCNN, self).__init__(
            batch=batch,
            use_glove=use_glove,
            train_embeddings=train_embeddings,
            embedding_path=embedding_path
        )

        self.units = units
        self.dropout_rate = dropout_rates
        self.params['units'] = units
        self.params['dropout_rates'] = dropout_rates

    def transform_embedded_sequences(self, embedded_sequences):
        x = Conv1D(self.units, 5, activation='relu', name='c1')(embedded_sequences)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu', name='c2')(x)
        x = MaxPooling1D(5)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = Conv1D(self.units, 5, activation='relu', name='c3', data_format='channels_first')(x)
        x = MaxPooling1D(35)(x)
        if self.dropout_rate > 0:
            x = Dropout(self.dropout_rate)(x)
        x = GlobalMaxPooling1D()(x)
        # x = Flatten()(x)
        x = Dense(self.units, activation='relu')(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds


class LSTMClassifier(KerasTextClassifier):

    def __init__(
            self,
            batch,
            use_glove,
            bidirectional=True,
            train_embeddings=False,
            embedding_path=None,
            units=128,
            layers=3,
            dropout_rates=0.25,
            rec_dropout_rate=0.2,
    ):
        super(LSTMClassifier, self).__init__(
            batch=batch,
            use_glove=use_glove,
            train_embeddings=train_embeddings,
            embedding_path=embedding_path
        )

        self.params['layers'] = layers
        self.params['units'] = units
        self.params['dropout_rates'] = dropout_rates
        self.params['rec_dropout_rate'] = rec_dropout_rate
        self.params['bidirectional'] = bidirectional

    def transform_embedded_sequences(self, embedded_sequences):
        x = embedded_sequences
        for i in range(1, self.params['layers'] + 1):
            # if there are more lstms downstream - return sequences
            return_sequences = i < self.params['layers']
            layer = LSTM(
                self.params['units'],
                dropout=self.params['dropout_rates'],
                recurrent_dropout=self.params['rec_dropout_rate'],
                return_sequences=return_sequences)
            if self.params['bidirectional']:
                x = Bidirectional(layer)(x)
            else:
                x = layer(x)
        preds = Dense(self.class_count, activation='softmax')(x)
        return preds


class YKimCNN(KerasTextClassifier):
    """Based on Alexander Rakhlin's implementation
    https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras
    of Yoon Kim's architecture from the paper
    https://arxiv.org/pdf/1408.5882v2.pdf
    """

    def __init__(
            self,
            batch,
            use_glove,
            train_embeddings=False,
            embedding_path=None,
            units=64,
            dropout_rates=(0.25, 0.5),
            filter_sizes=(3, 8),
            num_filters=10,
    ):
        super(YKimCNN, self).__init__(
            batch=batch,
            use_glove=use_glove,
            train_embeddings=train_embeddings,
            embedding_path=embedding_path
        )

        self.params['units'] = units
        self.params['dropout_rates'] = dropout_rates
        self.params['filter_sizes'] = filter_sizes
        self.params['num_filters'] = num_filters
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.dropout_rates = dropout_rates
        self.units = units

    def transform_embedded_sequences(self, embedded_sequences):
        drop_1, drop_2 = self.dropout_rates
        z = Dropout(drop_1)(embedded_sequences)

        conv_blocks = []
        for sz in self.filter_sizes:
            conv = Conv1D(
                filters=self.num_filters,
                kernel_size=sz,
                padding="valid",
                activation="relu",
                strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(drop_2)(z)
        z = Dense(self.units, activation="relu")(z)
        model_output = Dense(self.class_count, activation="softmax")(z)
        return model_output
