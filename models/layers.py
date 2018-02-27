from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.engine.topology import Layer
import keras.backend as K
from keras.layers.normalization import BatchNormalization


class PredictLayer(object):
    def __init__(self, dense_dim, input_dim=0, dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='relu',
                             input_shape=(input_dim,)))
        # self.model.add(Dense(dense_dim,
        #                      activation='relu'))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(3, activation='softmax'))

    def __call__(self, inputs):
        return self.model(inputs)


class HiddenStateBidirectional(Bidirectional):
    """
    When the inner RNN layer "return_state" is True, we'll also return the cell states.
    [h_fw, h_bw, c_fw, c_bw]
    Important for encoder-decoder application.
    """

    def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
        super(HiddenStateBidirectional, self).__init__(layer, merge_mode, weights, **kwargs)
        self.layer = layer
        self.h = None  # hidden state for forward
        self.h_rev = None  # hidden state for backward
        self.c = None  # cell state for forward
        self.c_rev = None  # cell state for backward)

    def compute_output_shape(self, input_shape):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward_layer.compute_output_shape(input_shape)
        elif self.merge_mode == 'concat':
            shape = list(self.forward_layer.compute_output_shape(input_shape))
            shape[-1] *= 2
            return tuple(shape)
        elif self.merge_mode is None:
            if self.h is not None \
                    and self.h_rev is not None \
                    and self.c is not None \
                    and self.c_rev is not None:
                return [self.forward_layer.compute_output_shape(input_shape)] * 6  # LSTM
            elif self.c is not None and self.c_rev is not None:
                return [self.forward_layer.compute_output_shape(input_shape)] * 4  # GRU
            else:
                return [self.forward_layer.compute_output_shape(input_shape)] * 2

    def call(self, inputs, training=None, mask=None):
        from keras.utils.generic_utils import has_arg
        kwargs = {}
        if has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if has_arg(self.layer.call, 'mask'):
            kwargs['mask'] = mask

        output = None  # output (i.e., hidden state)
        inputs_fw, inputs_bw = inputs, inputs

        if isinstance(inputs, (list, tuple)) and len(inputs) > 1:  # have customized initial hidden and cell state
            if len(inputs) == 5:  # LSTM
                inputs, h_fw, h_bw, c_fw, c_bw = inputs
                inputs_fw = [inputs, h_fw, c_fw]
                inputs_bw = [inputs, h_bw, c_bw]
            elif len(inputs) == 3:  # GRU
                inputs, h_fw, h_bw = inputs
                inputs_fw = [inputs, h_fw]
                inputs_bw = [inputs, h_bw]
            else:
                raise ValueError('inconsistent state number.')

        y = self.forward_layer.call(inputs_fw, **kwargs)
        y_rev = self.backward_layer.call(inputs_bw, **kwargs)

        if isinstance(y, (list, tuple)) and len(y) > 1:  # LSTM or GRU with return_state=True
            if isinstance(self.layer, LSTM):  # LSTM
                y, self.h, self.c = y
                y_rev, self.h_rev, self.c_rev = y_rev
            if isinstance(self.layer, GRU):  # GRU
                y, self.c = y
                y_rev, self.c_rev = y_rev

        if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == 'concat':
            output = K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            output = y + y_rev
        elif self.merge_mode == 'ave':
            output = (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            output = y * y_rev
        elif self.merge_mode is None:
            output = [y, y_rev]

        # Properly set learning phase
        if 0 < self.layer.dropout + self.layer.recurrent_dropout:
            if self.merge_mode is None:
                for out in output:
                    out._uses_learning_phase = True
            else:
                output._uses_learning_phase = True

        if self.h is not None \
                and self.h_rev is not None \
                and self.c is not None \
                and self.c_rev is not None:
            output = output + [self.h, self.h_rev, self.c, self.c_rev]  # LSTM o, o_rev, h, h_rev, c, c_rev
        elif self.c is not None and self.c_rev is not None:
            output = output + [self.c, self.c_rev]  # GRU o, o_rev, c, c_rev

        return output


class AspectAttentionLayer(Layer):
    def __init__(self, n_reason=5, hidden_d=100, **kwargs):
        self.n_reason = n_reason
        self.hidden_d = hidden_d
        super(AspectAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        embedding_size = input_shape[-1]
        self.W1 = self.add_weight(shape=(embedding_size, self.hidden_d),
                                  name='W1',
                                  initializer='glorot_uniform',
                                  trainable=True)
        # self.b1 = self.add_weight(shape=(self.hidden_d,),
        #                           initializer='zeros',
        #                           name='b1',
        #                           trainable=True)
        self.W2 = self.add_weight(shape=(self.hidden_d, self.n_reason),
                                  name='W2',
                                  initializer='glorot_uniform',
                                  trainable=True)
        # self.b2 = self.add_weight(shape=(self.n_reason,),
        #                           initializer='zeros',
        #                           name='b2',
        #                           trainable=True)
        self.built = True

    def call(self, inputs, **kwargs):
        H = inputs[0]
        A1 = K.tanh(K.dot(H, self.W1))
        A = K.softmax(K.dot(A1, self.W2))
        M = K.batch_dot(K.permute_dimensions(A, (0, 2, 1)), H)
        m_merge = K.max(M, axis=1)
        return m_merge

    # def call(self, inputs, **kwargs):
    #     H = inputs[0]
    #     sim = K.tanh(K.batch_dot(K.dot(H, self.W1), K.permute_dimensions(H, (0, 2, 1))))
    #     expanded_sim = K.expand_dims(sim, axis=-1)
    #     H = K.expand_dims(H, axis=1)
    #     weighted_sum = K.sum(expanded_sim * H, axis=2)
    #     sum_sim = K.expand_dims(K.sum(sim, axis=-1), axis=-1)
    #     attentive_vector = weighted_sum / sum_sim
    #     attentive_vector = K.max(attentive_vector, axis=1)
    #     return attentive_vector

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
        return input_shape[0], input_shape[2]

    def get_config(self):
        config = {'n_reason': self.n_reason,
                  'hidden_d': self.hidden_d}
        base_config = super(AspectAttentionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
