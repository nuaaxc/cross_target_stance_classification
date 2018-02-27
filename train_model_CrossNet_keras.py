import numpy as np
from keras import backend as K
import pickle
import glob
from config import (
    DirConfig,
    TrainConfig,
    TestConfig,
    CrossNetConfig as ModelConfig
)
# import tensorflow as tf
from data_util import (load_input_matrix)
from keras.callbacks import ModelCheckpoint, EarlyStopping
from models.CrossNet import build_model as build_crossnet

# np.random.seed(TrainConfig.SEED)
# print('random seed set.')
# tf.set_random_seed(TrainConfig.SEED)


def train_model(target, dir_config, train_config, model_config):
    print('###### Start training ######')
    print('------ target domain:', target)
    # load input matrix
    print('--- loading input matrix ...')
    # Load train/valid/test data set
    (train_x, train_t, train_labels, _,
     _, _, _, _, _, _,
     word_index, embedding_matrix) = load_input_matrix(target, dir_config, train_config)

    print('------ training data shape:')
    print(train_x.shape)
    print(train_t.shape)
    print(train_labels.shape)

    print('------ embedding matrix shape:')
    print(embedding_matrix.shape)

    # Build model
    model = build_crossnet(embedding_matrix, word_index, train_config, model_config)

    # Define model callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model_file = model_config.BASE_DIR + '%s_rnn_%s_seq_%d_context_%d_dense_%d_R_%d_drop_%.2f_target_%s' % \
                                         (model_config.MODEL, model_config.RNN_UNIT, train_config.MAX_SEQUENCE_LENGTH,
                                          model_config.LSTM_DIM, model_config.DENSE_DIM, model_config.R,
                                          model_config.DROP_RATE, target)
    model_checkpoint = ModelCheckpoint(model_file + '_model.h5', save_best_only=True, save_weights_only=True)

    # Training
    history = model.fit(
        x=[train_x, train_t],
        y=train_labels,
        validation_split=0.2,
        epochs=TrainConfig.NB_EPOCH,
        batch_size=TrainConfig.BATCH_SIZE,
        callbacks=[early_stopping, model_checkpoint],
        shuffle=True,
        verbose=2)

    K.clear_session()


def test_model(target, dir_config, train_config, test_config, model_config):
    print('###### Start testing ######')
    print('------ target domain:', target)

    # Load test data
    print('--- loading input matrix ...')
    (_, _, _, _,
     test_x, test_t, test_labels, test_target_ind, test_id, test_text,
     word_index, embedding_matrix) = load_input_matrix(target, dir_config, train_config)
    print('------ test data shape:')
    print(test_x.shape)
    print(test_t.shape)
    print(test_target_ind.shape)
    print(test_labels.shape)
    print(test_id.shape)
    print(test_text.shape)

    # Load models from cache
    print('--- loading model ...')
    # model = load_keras_model(model_config, target)
    model_weight_path = glob.glob(model_config.BASE_DIR + '%s*context_%d_dense_%d_R_%s_*target_%s_model.h5'
                                  % (model_config.MODEL, model_config.LSTM_DIM,
                                     model_config.DENSE_DIM, model_config.R, target))[0]
    model = build_crossnet(embedding_matrix, word_index, train_config, model_config)
    model.load_weights(model_weight_path)

    print('--- predicting ...')
    # Testing
    preds = model.predict(
        [test_x, test_t],
        batch_size=test_config.BATCH_SIZE,
        verbose=1)

    K.clear_session()

    pred_labels = [dir_config.LABEL_MAPPING_INV[np.argmax(label)] for label in preds]
    test_labels = [dir_config.LABEL_MAPPING_INV[np.argmax(label)] for label in test_labels]

    if 'dt' in target:
        pred_labels = []
        for i in range(len(preds)):
            label = preds[i]
            text = test_text[i].lower()
            if 'trump' in text or 'donald' in text:
                new_label = dir_config.LABEL_MAPPING_INV[np.argmax(label[:2])]
            else:
                new_label = dir_config.LABEL_MAPPING_INV[np.argmax(label)]
            pred_labels.append(new_label)

    from sklearn import metrics
    f1_weighted = metrics.f1_score(test_labels, pred_labels, average='weighted')
    f1_micro = metrics.f1_score(test_labels, pred_labels, average='micro')
    fi_macro = metrics.f1_score(test_labels, pred_labels, average='macro')
    accuracy = metrics.accuracy_score(test_labels, pred_labels)
    print('f1-score (weighted):', f1_weighted)
    print('f1-score (micro):', f1_micro)
    print('f1-score (macro):', fi_macro)
    print('accuracy:', accuracy)
    return f1_weighted, f1_micro, fi_macro, accuracy


def main_cmd(args):
    """
    -tr_te --target cc_cc --n_reason 1 --bsize 256 --max_epoch 100
    """
    np.random.seed(TrainConfig.SEED)
    TrainConfig.BATCH_SIZE = args.bsize
    TrainConfig.NB_EPOCH = args.max_epoch
    ModelConfig.R = args.n_reason

    if args.train is False and args.test is False and args.tr_te is False:
        print('Please specify either "-train" mode or "-test" mode or "tr_te" mode.')
        return
    if args.train:
        train_model(args.target, DirConfig, TrainConfig, ModelConfig)
    if args.test:
        test_model(args.target, DirConfig, TrainConfig, TestConfig, ModelConfig)
    if args.tr_te:
        train_model(args.target, DirConfig, TrainConfig, ModelConfig)
        test_model(args.target, DirConfig, TrainConfig, TestConfig, ModelConfig)


def main_all(num):
    TrainConfig.BATCH_SIZE = 128
    TrainConfig.NB_EPOCH = 100
    ModelConfig.R = 1
    result = {}
    for target_tr, target_te in [('fm', 'la'), ('la', 'fm'), ('hc', 'dt'), ('dt', 'hc')]:
        res = {}
        target = target_tr + '_' + target_te
        train_model(target, DirConfig, TrainConfig, ModelConfig)
        f1_weighted, f1_micro, fi_macro, accuracy = \
            test_model(target, DirConfig, TrainConfig, TestConfig, ModelConfig)
        res[target_te] = {
            'weighted': f1_weighted,
            'micro': f1_micro,
            'macro': fi_macro,
            'accuracy': accuracy
        }
        result[target_tr] = res
    print('saving to pickle ...')
    pickle.dump(result, open(DirConfig.RESULT_PAIRWISE % (ModelConfig.MODEL, num), 'wb'))
    print('result saved.')


def main_single(target):
    TrainConfig.BATCH_SIZE = 128
    TrainConfig.NB_EPOCH = 100
    ModelConfig.R = 1
    train_model(target, DirConfig, TrainConfig, ModelConfig)
    f1_weighted, f1_micro, fi_macro, accuracy = \
        test_model(target, DirConfig, TrainConfig, TestConfig, ModelConfig)
    print('F:', (f1_micro + fi_macro) * 0.5)


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser(description='Stance classification')
    # parser.add_argument('-tr_te', action='store_true', help='train and test model')
    # parser.add_argument('-train', action='store_true', help='train model')
    # parser.add_argument('-test', action='store_true', help='test model')
    # parser.add_argument('--max_epoch', type=int, default=100, help='max epoch number')
    # parser.add_argument('--bsize', type=int, default=64, help='batch size')
    # parser.add_argument('--n_reason', type=int, default=5, help='batch size')
    # parser.add_argument('--target', type=str, default='all', help='target domain for train')

    # args = parser.parse_args()

    # for i in range(1, 21):
    #     main_all(i)

    main_single('hc_dt')
    # main_single('fm_la')
    # main_single('la_fm')
    # main_single('dt_hc')
    # main_single('cc_amp')
    pass
