# -*- coding: utf-8 -*-
import keras.backend as K
from keras import optimizers, regularizers, losses
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dropout
from keras.models import Model
from models.layers import (
    PredictLayer,
    AspectAttentionLayer,
    AspectEncoding,
    LayerNormalization
)


def build_model(embedding_matrix, word_index, train_config, model_config, dir_config):
    print('--- Building model...')

    # Parameters
    sent_length = train_config.MAX_SENT_LENGTH
    target_length = train_config.MAX_TARGET_LENGTH
    nb_words = min(train_config.MAX_NB_WORDS, len(word_index)) + 1
    word_embedding_dim = train_config.WORD_EMBEDDING_DIM
    dropout_rate = model_config.DROP_RATE
    rnn_dim = model_config.RNN_DIM
    n_aspect = model_config.NUM_ASPECT
    dense_dim = model_config.DENSE_DIM
    lr = train_config.LR
    num_class = len(dir_config.LABEL_MAPPING)

    # Input layer
    sent = Input(shape=(sent_length,), dtype='int32', name='s_input')
    target = Input(shape=(target_length,), dtype='int32', name='t_input')

    # Embedding Layer
    emb_sent = Embedding(output_dim=word_embedding_dim,
                         input_dim=nb_words,
                         input_length=sent_length,
                         weights=[embedding_matrix],
                         trainable=False,
                         mask_zero=True)(sent)
    emb_target = Embedding(output_dim=word_embedding_dim,
                           input_dim=nb_words,
                           input_length=target_length,
                           weights=[embedding_matrix],
                           trainable=False,
                           mask_zero=True)(target)

    emb_sent = Dropout(dropout_rate)(emb_sent)
    emb_target = Dropout(dropout_rate)(emb_target)

    # Context Encoding Layer
    target_encoding_layer = Bidirectional(LSTM(rnn_dim,
                                               dropout=dropout_rate,
                                               recurrent_dropout=dropout_rate,
                                               return_state=True,
                                               return_sequences=False),
                                          merge_mode='concat')
    (target_encoding,
     target_fw_state_h, target_fw_state_s,
     target_bw_state_h, target_bw_state_s) = target_encoding_layer(emb_target)

    sent_encoding_layer = Bidirectional(LSTM(rnn_dim,
                                             unroll=True,
                                             kernel_regularizer=regularizers.l2(1e-4),
                                             activity_regularizer=regularizers.l2(1e-4),
                                             dropout=dropout_rate,
                                             recurrent_dropout=dropout_rate,
                                             return_state=False,
                                             return_sequences=True),
                                        merge_mode='concat')
    sent_encoding = sent_encoding_layer(emb_sent,
                                        initial_state=[target_fw_state_h, target_fw_state_s,
                                                       target_bw_state_h, target_bw_state_s])

    # Aspect Attention Layer
    aspect_attention_layer = AspectAttentionLayer(n_aspect=n_aspect, hidden_d=dense_dim)
    aspect_attention = aspect_attention_layer([sent_encoding])

    # Aspect Encoding Layer
    aspect_encoding_layer = AspectEncoding()
    aspect_encoding = aspect_encoding_layer([aspect_attention, sent_encoding])

    aspect_encoding = LayerNormalization()(aspect_encoding)

    # Prediction layer
    pred = PredictLayer(dense_dim,
                        input_dim=K.int_shape(aspect_encoding)[-1],
                        dropout=dropout_rate,
                        num_class=num_class)(aspect_encoding)

    # Build model graph
    model = Model(inputs=(sent, target),
                  outputs=pred)

    # Compile model
    optimizer = optimizers.Nadam(lr=lr)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model
