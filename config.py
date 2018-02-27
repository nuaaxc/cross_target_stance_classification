# -*- coding: utf-8 -*-
"""
Model configs.
"""
import os


class DirConfig(object):
    home = str(os.path.expanduser('~'))
    DATA_DIR = ''
    if 'nuaax' in home:
        DATA_DIR = 'C:/Users/nuaax/Dropbox/data61/project/stance_classification/dataset/semeval/'
    elif 'xu052' in home:
        DATA_DIR = '/home/xu052/stance/dataset/semeval/'
    W2V_FILE = 'C:/Users/nuaax/Downloads/dataset/word_vec/GoogleNews-vectors-negative300.bin'
    GLOVE_FILE = 'C:/Users/nuaax/Downloads/dataset/word_vec/glove.twitter.27B.200d.txt'

    CODE_TARGET = {
        'cc': 'Climate Change is a Real Concern',
        'a': 'Atheism',
        'fm': 'Feminist Movement',
        'hc': 'Hillary Clinton',
        'la': 'Legalization of Abortion',
        'dt': 'Donald Trump',
        'amp': 'AMP',
    }
    TARGET_INDEX = {
        'Climate Change is a Real Concern': 0,
        'Atheism': 1,
        'Feminist Movement': 2,
        'Hillary Clinton': 3,
        'Legalization of Abortion': 4,
        'Donald Trump': 5,
        'AMP': 6,
    }
    TARGET_INDEX_INV = {
        0: 'Climate Change is a Real Concern',
        1: 'Atheism',
        2: 'Feminist Movement',
        3: 'Hillary Clinton',
        4: 'Legalization of Abortion',
        5: 'Donald Trump',
        6: 'AMP',
    }
    TARGETS = ['cc', 'a', 'fm', 'hc', 'la']
    TARGET_NUM = len(TARGET_INDEX)
    TRAIN_FILE = DATA_DIR + 'semeval2016-task6-train+dev_%s.txt'
    TEST_FILE = DATA_DIR + 'SemEval2016-Task6-subtaskA-testdata-gold_%s.txt'
    TEST_DT_FILE = DATA_DIR + 'SemEval2016-Task6-subtaskB-testdata-gold.txt'
    AMP_RAW_DATA = DATA_DIR + 'adani_esa_till_201706_tweets.json'
    TEST_AMP_FILE = DATA_DIR + 'SemEval2016-Task6-subtaskA-testdata-gold_amp.txt'
    # cache
    CACHE_DIR = DATA_DIR + 'cache/%s/'
    CACHE_TRAIN = CACHE_DIR + 'cache_train.npy'
    CACHE_TRAIN_TARGET = CACHE_DIR + 'cache_train_target.npy'
    CACHE_TRAIN_TARGET_INDEX = CACHE_DIR + 'cache_train_target_ind.npy'
    CACHE_TRAIN_LABEL = CACHE_DIR + 'cache_train_label.npy'

    CACHE_VALID = CACHE_DIR + 'cache_valid.npy'
    CACHE_VALID_TARGET = CACHE_DIR + 'cache_valid_target.npy'
    CACHE_VALID_TARGET_INDEX = CACHE_DIR + 'cache_valid_target_ind.npy'
    CACHE_VALID_LABEL = CACHE_DIR + 'cache_valid_label.npy'

    CACHE_TEST = CACHE_DIR + 'cache_test.npy'
    CACHE_TEST_TARGET = CACHE_DIR + 'cache_test_target.npy'
    CACHE_TEST_TARGET_INDEX = CACHE_DIR + 'cache_test_target_ind.npy'
    CACHE_TEST_LABEL = CACHE_DIR + 'cache_test_label.npy'
    CACHE_TEST_ID = CACHE_DIR + 'cache_test_id.npy'
    CACHE_TEST_TEXT = CACHE_DIR + 'cache_test_text.npy'

    W2V_CACHE = CACHE_DIR + 'w2v_matrix.npy'
    GLOVE_CACHE = CACHE_DIR + 'glove_matrix.npy'
    WORD_INDEX_CACHE = CACHE_DIR + 'word_index.npy'
    LABEL_MAPPING = {'FAVOR': 0, 'AGAINST': 1, 'NONE': 2}
    LABEL_MAPPING_INV = {0: 'FAVOR', 1: 'AGAINST', 2: 'NONE'}

    RESULT_PAIRWISE = DATA_DIR + 'result_pairwise_%s_%d.pkl'
    FIG_PAIRWISE = DATA_DIR + 'fig_pairwise_%s_%s.png'


class TrainConfig(object):
    SEED = 2018
    LR = 0.001
    TEST_SIZE = 0.1
    RE_WEIGHT = True
    BATCH_SIZE = 128
    NB_EPOCH = 2
    REMOVE_STOPWORDS = 1
    USE_STEM = 0
    W2V_TYPE = 'glove'
    MAX_SEQUENCE_LENGTH = 20
    MAX_TARGET_LENGTH = 4
    MAX_NB_WORDS = 200000
    WORD_EMBEDDING_DIM = 200
    VALIDATION_SPLIT = 0.1


class TestConfig(object):
    RE_WEIGHT = True
    BATCH_SIZE = 64


class CrossNetConfig(TrainConfig):
    MODEL = 'CrossNet'
    RNN_UNIT = 'LSTM'
    BASE_DIR = DirConfig.DATA_DIR + 'models/'
    LSTM_DIM = 100
    DENSE_DIM = 300
    DROP_RATE = 0.1
    R = 1


class TOPConfig(TrainConfig):
    MODEL = 'Top'
    RNN_UNIT = 'LSTM'
    BASE_DIR = DirConfig.DATA_DIR + 'models/'
    LSTM_DIM = 100
    DENSE_DIM = 300
    DROP_RATE = 0.1


class BiLSTMConfig(TrainConfig):
    MODEL = 'BiLSTM'
    RNN_UNIT = 'LSTM'
    BASE_DIR = DirConfig.DATA_DIR + 'models/'
    LSTM_DIM = 100
    DENSE_DIM = 300
    DROP_RATE = 0.1
