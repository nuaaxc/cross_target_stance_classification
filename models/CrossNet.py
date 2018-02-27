# -*- coding: utf-8 -*-
import keras.backend as K
from keras import optimizers
from keras.layers import Input, Embedding, LSTM
from keras.models import Model
from models.layers import (
    PredictLayer,
    HiddenStateBidirectional,
    AspectAttentionLayer,
)


def build_model(embedding_matrix, word_index, train_config, model_config):
    print('--- Building model...')

    # Parameters
    sequence_length = train_config.MAX_SEQUENCE_LENGTH
    target_length = train_config.MAX_TARGET_LENGTH
    nb_words = min(train_config.MAX_NB_WORDS, len(word_index)) + 1
    word_embedding_dim = train_config.WORD_EMBEDDING_DIM
    dropout = model_config.DROP_RATE
    rnn_dim = model_config.LSTM_DIM
    n_reason = model_config.R
    dense_dim = model_config.DENSE_DIM
    lr = train_config.LR

    # Input layer
    s = Input(shape=(sequence_length,), dtype='int32', name='s_input')
    t = Input(shape=(target_length,), dtype='int32', name='t_input')

    # Embedding Layer
    s_rep = Embedding(output_dim=word_embedding_dim,
                      input_dim=nb_words,
                      input_length=sequence_length,
                      weights=[embedding_matrix],
                      trainable=False)(s)
    t_rep = Embedding(output_dim=word_embedding_dim,
                      input_dim=nb_words,
                      input_length=target_length,
                      weights=[embedding_matrix],
                      trainable=False)(t)

    # Context Encoding Layer
    target_context = HiddenStateBidirectional(LSTM(rnn_dim,
                                                   dropout=dropout,
                                                   recurrent_dropout=dropout,
                                                   return_state=True,
                                                   return_sequences=False),
                                              merge_mode=None,
                                              input_shape=(target_length, K.int_shape(t_rep)[-1],))
    _, _, t_h_fw, t_h_bw, t_c_fw, t_c_bw = target_context(t_rep)

    context_encoding_layer = HiddenStateBidirectional(LSTM(rnn_dim,
                                                           unroll=True,
                                                           dropout=dropout,
                                                           recurrent_dropout=dropout,
                                                           return_state=False,
                                                           return_sequences=True),
                                                      merge_mode='concat',
                                                      input_shape=(sequence_length, K.int_shape(s_rep)[-1],))
    sent_context = context_encoding_layer([s_rep, t_h_fw, t_h_bw, t_c_fw, t_c_bw])

    # Aspect Attention Layer
    aspect_attention_layer = AspectAttentionLayer(n_reason=n_reason, hidden_d=dense_dim)
    aspect_repr = aspect_attention_layer([sent_context])

    # Prediction layer
    pred = PredictLayer(dense_dim,
                        input_dim=K.int_shape(aspect_repr)[-1],
                        dropout=dropout)(aspect_repr)

    # Build model graph
    model = Model(inputs=(s, t),
                  outputs=pred)

    # Compile model
    nadam = optimizers.Nadam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=nadam,
                  metrics=['accuracy'])
    # model.summary()
    return model
