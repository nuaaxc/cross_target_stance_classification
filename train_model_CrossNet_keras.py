import numpy as np
from keras import backend as K
from config import (
    DirConfig,
    TrainConfig,
    CrossNetConfig as ModelConfig
)

from data_util import (load_input_matrix)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.CrossNet import build_model as build_crossnet

# import tensorflow as tf
# np.random.seed(TrainConfig.SEED)
# print('random seed set.')
# tf.set_random_seed(TrainConfig.SEED)


def train_model(target, dir_config, train_config, model_config):
    print('###### Start training ######')
    print('------ target domain:', target)
    # load input matrix
    print('--- loading input matrix ...')
    # Load train/valid/test data set
    (train_x, train_t, train_labels,
     _, _, _, _, _,
     word_index, embedding_matrix) = load_input_matrix(target, dir_config, train_config)

    print('------ training data shape:')
    print(train_x.shape)
    print(train_t.shape)
    print(train_labels.shape)

    print('------ embedding matrix shape:')
    print(embedding_matrix.shape)

    # Build model
    model = build_crossnet(embedding_matrix, word_index, train_config, model_config, dir_config)

    # Define model callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    model_file_path = model_config.BASE_DIR + '%s_rnn_%s_seq_%d_context_%d_dense_%d_R_%d_drop_%.2f_lr_%f_target_%s_model.h5' % \
                      (model_config.MODEL, model_config.RNN_UNIT, train_config.MAX_SENT_LENGTH,
                       model_config.RNN_DIM, model_config.DENSE_DIM, model_config.NUM_ASPECT,
                       model_config.DROP_RATE, model_config.LR, target)
    model_checkpoint = ModelCheckpoint(model_file_path, save_best_only=True, save_weights_only=True)

    # Training
    model.fit(
        x=[train_x, train_t],
        y=train_labels,
        validation_split=train_config.VALIDATION_SPLIT,
        epochs=TrainConfig.NB_EPOCH,
        batch_size=TrainConfig.BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=True,
        verbose=2)

    K.clear_session()


def test_model(target, dir_config, train_config, model_config):
    print('###### Start testing ######')
    print('------ target domain:', target)

    # Load test data
    print('--- loading input matrix ...')
    (_, _, _,
     test_x, test_t, test_labels, test_id, test_text,
     word_index, embedding_matrix) = load_input_matrix(target, dir_config, train_config)
    print('------ test data shape:')
    print(test_x.shape)
    print(test_t.shape)
    print(test_labels.shape)
    print(test_id.shape)
    print(test_text.shape)

    # Load models from cache
    print('--- loading model ...')

    model_weight_path = model_config.BASE_DIR + '%s_rnn_%s_seq_%d_context_%d_dense_%d_R_%d_drop_%.2f_lr_%f_target_%s_model.h5' % \
                                  (model_config.MODEL, model_config.RNN_UNIT, train_config.MAX_SENT_LENGTH,
                                   model_config.RNN_DIM, model_config.DENSE_DIM, model_config.NUM_ASPECT,
                                   model_config.DROP_RATE, model_config.LR, target)
    print('Loading model from %s ...' % model_weight_path)
    model = build_crossnet(embedding_matrix, word_index, train_config, model_config, dir_config)
    model.load_weights(model_weight_path)

    print('--- predicting ...')
    # Testing
    preds = model.predict(
        [test_x, test_t],
        batch_size=train_config.BATCH_SIZE,
        verbose=1)

    K.clear_session()

    pred_labels = [dir_config.LABEL_MAPPING_INV[np.argmax(label)] for label in preds]
    test_labels = [dir_config.LABEL_MAPPING_INV[np.argmax(label)] for label in test_labels]

    from sklearn import metrics
    f1_weighted = metrics.f1_score(test_labels, pred_labels, average='weighted')
    f1_micro = metrics.f1_score(test_labels, pred_labels, average='micro')
    fi_macro = metrics.f1_score(test_labels, pred_labels, average='macro')
    accuracy = metrics.accuracy_score(test_labels, pred_labels)

    print(metrics.classification_report(test_labels, pred_labels))
    print('f1-score (weighted):', f1_weighted)
    print('f1-score (micro):', f1_micro)
    f1_favor, f1_against, _ = metrics.f1_score(test_labels, pred_labels, average=None)
    print('f1-score (macro, favor & against):', 0.5 * (f1_favor + f1_against))
    print('f1-score (macro, all 3 classes):', fi_macro)
    print('accuracy:', accuracy)
    return f1_weighted, f1_micro, fi_macro, accuracy


def main(args):
    np.random.seed(TrainConfig.SEED)
    TrainConfig.BATCH_SIZE = args.bsize
    TrainConfig.NB_EPOCH = args.max_epoch
    TrainConfig.LR = args.learning_rate
    ModelConfig.NUM_ASPECT = args.n_aspect
    ModelConfig.RNN_DIM = args.rnn_dim
    ModelConfig.DENSE_DIM = args.dense_dim
    ModelConfig.DROP_RATE = args.dropout_rate

    if args.train is False and args.test is False and args.tr_te is False:
        print('Please specify one of the modes: "-train", "-test", or "tr_te".')
        return
    if args.train:
        train_model(args.target, DirConfig, TrainConfig, ModelConfig)
    if args.test:
        test_model(args.target, DirConfig, TrainConfig, ModelConfig)
    if args.tr_te:
        train_model(args.target, DirConfig, TrainConfig, ModelConfig)
        test_model(args.target, DirConfig, TrainConfig, ModelConfig)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Stance classification')
    parser.add_argument('-tr_te', action='store_true', help='train and test mode')
    parser.add_argument('-train', action='store_true', help='train mode')
    parser.add_argument('-test', action='store_true', help='test mode')
    parser.add_argument('--max_epoch', type=int, default=100, help='max epoch number')
    parser.add_argument('--bsize', type=int, default=64, help='batch size')
    parser.add_argument('--n_aspect', type=int, default=1, help='number of aspect')
    parser.add_argument('--target', type=str, default='cc_cc', help='target domain')
    parser.add_argument('--rnn_dim', type=int, default=256, help='RNN hidden size')
    parser.add_argument('--dense_dim', type=int, default=128, help='Dense hidden size')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    main(args)
