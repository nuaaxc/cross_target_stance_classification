from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from keras.layers.normalization import BatchNormalization


class AspectAttentionLayer(Layer):
    def __init__(self, n_aspect=5, hidden_d=100, **kwargs):
        self.n_aspect = n_aspect
        self.hidden_d = hidden_d
        self.supports_masking = True
        super(AspectAttentionLayer, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1]
        self.W1 = self.add_weight(shape=(embedding_size, self.hidden_d),
                                  name='W1',
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(shape=(self.hidden_d, self.n_aspect),
                                  name='W2',
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.built = True

    def call(self, inputs, mask=None):
        H = inputs[0]
        weight = K.softmax(K.dot(K.relu(K.dot(H, self.W1)), self.W2))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            weight = weight * K.expand_dims(mask)
            weight = weight / K.sum(weight, axis=1, keepdims=True)
        return K.permute_dimensions(weight, (0, 2, 1))  # batch_size, n_aspect, sent_length

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape[0], self.n_aspect, input_shape[1]

    def get_config(self):
        config = {'n_aspect': self.n_aspect,
                  'hidden_d': self.hidden_d}
        base_config = super(AspectAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AspectEncoding(Layer):
    def __init__(self, **kwargs):
        super(AspectEncoding, self).__init__(**kwargs)

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def build(self, input_shape):
        self.batch_size = input_shape[0][0]
        self.n_aspect = input_shape[0][1]
        self.sequence_length = input_shape[0][2]
        self.hidden_size = input_shape[1][2]
        super(AspectEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A, H = inputs
        M = K.batch_dot(A, H)
        M = K.max(M, axis=1)
        return M

    def compute_output_shape(self, input_shape):
        return self.batch_size, self.hidden_size


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x, **kwargs):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class PredictLayer(object):
    def __init__(self, dense_dim, input_dim=0, dropout=0.0, num_class=3):
        self.model = Sequential()
        self.model.add(Dense(dense_dim, activation='relu', input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(Dense(dense_dim, activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(num_class, activation='softmax'))

    def __call__(self, inputs):
        return self.model(inputs)
