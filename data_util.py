import re
from tokenizer.twokenize_wrapper import tokenize
from nltk.corpus import stopwords
from config import (
    DirConfig,
    TrainConfig
)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.utils import to_categorical
import pandas as pd
import json
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors
import os


def get_word_seq(train, train_target, test, test_target):
    # fit tokenizer
    tk = Tokenizer(num_words=TrainConfig.MAX_NB_WORDS)
    tk.fit_on_texts(train + train_target + test + test_target)
    word_index = tk.word_index

    # training text sequence (input matrix: shape - [sentence_len, MAX_SENT_LENGTH])
    train_x, train_t = '', ''
    if len(train) > 0 and len(train_target) > 0:
        train_x = tk.texts_to_sequences(train)
        train_x = pad_sequences(train_x, maxlen=TrainConfig.MAX_SENT_LENGTH)
        train_t = tk.texts_to_sequences(train_target)
        train_t = pad_sequences(train_t, maxlen=TrainConfig.MAX_TARGET_LENGTH)

    # testing text sequence
    test_x = tk.texts_to_sequences(test)
    test_x = pad_sequences(test_x, maxlen=TrainConfig.MAX_SENT_LENGTH)
    test_t = tk.texts_to_sequences(test_target)
    test_t = pad_sequences(test_t, maxlen=TrainConfig.MAX_TARGET_LENGTH)

    return train_x, train_t, test_x, test_t, word_index


def filter_stopwords(tokenised_tweet, filt='all'):
    """
    Remove stopwords from tokenised tweet
    :param filt:
    :param tokenised_tweet: tokenised tweet
    :return: tweet tokens without stopwords
    """
    if filt == "all":
        stops = stopwords.words("english")
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filt == "most":
        stops = []
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~", "=", "+", "!", "?"])
        stops.extend(["rt", "#semst", "...", "thats", "im", "'s", "via"])
    elif filt == "punctonly":
        stops = []
        # extended with string.punctuation and rt and #semst, removing links further down
        stops.extend(["\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", ",", "-", ".", "/", ":",
                      ";", "<", ">", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"])  # "=", "+", "!",  "?"
        stops.extend(["rt", "#semst", "..."])  # "thats", "im", "'s", "via"])
    else:
        stops = ["rt", "#semst", "..."]

    stops = set(stops)
    return [w for w in tokenised_tweet if (w not in stops and not w.startswith("http"))]


def text_to_wordlist(text):
    filtered = filter_stopwords(tokenize(text.lower()))
    return " ".join(filtered)


def preprocess_texts(texts, is_target=False):
    processed = []
    for text in texts:
        if is_target:
            processed.append(str(text).lower())  # deal with target words
        else:
            processed.append(text_to_wordlist(text))
    return processed


def load_word_embedding(which, vec_file):
    if which == 'glove':
        return load_glove_matrix(vec_file)
    else:
        return load_word2vec_matrix(vec_file)


def load_word2vec_matrix(vec_file):
    return KeyedVectors.load_word2vec_format(vec_file, binary=True)


def load_glove_matrix(vec_file):
    word2vec = {}
    with open(vec_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            word2vec[word] = coefs
    print('Found %s word vectors.' % len(word2vec))
    return word2vec


def save_glove_matrix(word2vec, word_index, output_file, config):
    nb_words = min(config.MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, config.WORD_EMBEDDING_DIM))
    for word, i in tqdm(word_index.items()):
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        # else:
        #     embedding_matrix[i] = np.random.rand(config.WORD_EMBEDDING_DIM)

    print('Vocabulary size: %d' % len(word_index))
    print('Valid word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    print('saving glove matrix: %s ...' % output_file)
    np.save(output_file, embedding_matrix)
    print('saved.')


def save_word2vec_matrix(word2vec, word_index, output_file, config):
    nb_words = min(config.MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((nb_words, config.WORD_EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)
    print('Vocabulary size: %d' % len(word_index))
    print('Valid word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) != 0))
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    print('saving word2vec matrix: %s ...' % output_file)
    np.save(output_file, embedding_matrix)
    print('saved.')


def save_training_history(history_file, history):
    results = pd.DataFrame(data=history.history)
    results.to_csv(history_file + '_history.csv')
    print('--- Saved training history.')


def save_model(model_file, model):
    model.save(model_file + '_model.h5')
    print('--- Saved model.')


def load_keras_model(model_config, target):
    import glob
    model_path = glob.glob(model_config.BASE_DIR + '%s*R_%s_*target_%s_model.h5'
                           % (model_config.MODEL, model_config.R, target))[0]
    print('----- model:', model_path)
    model = load_model(model_path)
    return model


def text2matrix(target_tr, target_te, word2vec, encoding):
    """
    convert text data (train/validate/test) to matrix
    """
    target = target_tr + '_' + target_te
    # load data
    print('--- loading raw data ...')

    if target_tr != target_te:
        train_data_tr = None
        if os.path.exists(DirConfig.TRAIN_FILE % target_tr):
            print('------ train:', DirConfig.TRAIN_FILE % target_tr)
            train_data_tr = pd.read_csv(DirConfig.TRAIN_FILE % target_tr, encoding=encoding, sep='\t')
        print('------ train:', DirConfig.TEST_FILE % target_tr)
        train_data_te = pd.read_csv(DirConfig.TEST_FILE % target_tr, encoding=encoding, sep='\t')
        train_data = pd.concat([train_data_tr, train_data_te])

        test_data_tr = None
        if os.path.exists(DirConfig.TRAIN_FILE % target_te):
            print('------ test:', DirConfig.TRAIN_FILE % target_te)
            test_data_tr = pd.read_csv(DirConfig.TRAIN_FILE % target_te, encoding=encoding, sep='\t')
        print('------ test:', DirConfig.TEST_FILE % target_te)
        test_data_te = pd.read_csv(DirConfig.TEST_FILE % target_te, encoding=encoding, sep='\t')
        test_data = pd.concat([test_data_tr, test_data_te])
    elif target_tr != 'dt':
        print('------ train:', DirConfig.TRAIN_FILE % target_tr)
        train_data = pd.read_csv(DirConfig.TRAIN_FILE % target_tr, encoding=encoding, sep='\t')
        print('------ test:', DirConfig.TEST_FILE % target_te)
        test_data = pd.read_csv(DirConfig.TEST_FILE % target_te, encoding=encoding, sep='\t')
    else:
        train_data = pd.read_csv(DirConfig.TEST_FILE % target_tr, encoding=encoding, sep='\t')
        test_data = pd.read_csv(DirConfig.TEST_FILE % target_te, encoding=encoding, sep='\t')

    # train text
    train = list(train_data.Tweet.values.astype(str))
    train_target = list(train_data.Target.values.astype(str))
    train_labels = to_categorical(np.array([DirConfig.LABEL_MAPPING[label] for label in train_data.Stance.values]))

    assert np.all([t == DirConfig.CODE_TARGET[target_tr] for t in train_target])

    # test text
    test = list(test_data.Tweet.values.astype(str))
    test_target = list(test_data.Target.values.astype(str))
    test_labels = to_categorical(np.array([DirConfig.LABEL_MAPPING[label] for label in test_data.Stance.values]))
    test_id = list(test_data.ID.values.astype(str))
    test_text = list(test_data.Tweet.values.astype(str))

    assert np.all([t == DirConfig.CODE_TARGET[target_te] for t in test_target])

    # pre-process train/valid/test text
    print('--- preprocessing text ...')
    train = preprocess_texts(train)
    train_target = preprocess_texts(train_target, is_target=True)

    test = preprocess_texts(test)
    test_target = preprocess_texts(test_target, is_target=True)

    # convert into sequence
    print('--- converting into sequence ...')
    train_x, train_t, test_x, test_t, word_index = get_word_seq(train, train_target, test, test_target)

    # cache train, valid, test input matrix
    print('------ save ...')
    if not os.path.exists(DirConfig.CACHE_DIR % target):
        os.makedirs(DirConfig.CACHE_DIR % target)
    print('------ saving training matrix ...')
    np.save(DirConfig.CACHE_TRAIN % target, train_x)
    np.save(DirConfig.CACHE_TRAIN_TARGET % target, train_t)
    np.save(DirConfig.CACHE_TRAIN_LABEL % target, train_labels)

    print('------ saving test matrix ...')
    np.save(DirConfig.CACHE_TEST % target, test_x)
    np.save(DirConfig.CACHE_TEST_TARGET % target, test_t)
    np.save(DirConfig.CACHE_TEST_LABEL % target, test_labels)
    np.save(DirConfig.CACHE_TEST_ID % target, test_id)
    np.save(DirConfig.CACHE_TEST_TEXT % target, test_text)

    print('------ saving word_index ...')
    np.save(DirConfig.WORD_INDEX_CACHE % target, word_index)

    # save word embedding
    print('------ saving word2vec ...')
    save_glove_matrix(word2vec,
                      word_index,
                      DirConfig.GLOVE_CACHE % target,
                      TrainConfig)
    print('------ saved.')


def load_input_matrix(target, dir_config, train_config):
    print('------ training input matrix ...')
    train_x = np.load(open(dir_config.CACHE_TRAIN % target, 'rb'))
    train_t = np.load(open(dir_config.CACHE_TRAIN_TARGET % target, 'rb'))
    train_labels = np.load(open(dir_config.CACHE_TRAIN_LABEL % target, 'rb'))

    print('------ test input matrix ...')
    test_x = np.load(open(dir_config.CACHE_TEST % target, 'rb'))
    test_t = np.load(open(dir_config.CACHE_TEST_TARGET % target, 'rb'))
    test_labels = np.load(open(dir_config.CACHE_TEST_LABEL % target, 'rb'))
    test_id = np.load(open(dir_config.CACHE_TEST_ID % target, 'rb'))
    test_text = np.load(open(dir_config.CACHE_TEST_TEXT % target, 'rb'))

    print('------ word index ...')
    word_index = np.load(open(dir_config.WORD_INDEX_CACHE % target, 'rb')).item()

    print('------ embedding matrix ...')
    if train_config.W2V_TYPE == 'glove':
        embedding_matrix = np.load(open(dir_config.GLOVE_CACHE % target, 'rb'))
    else:
        embedding_matrix = np.load(open(dir_config.W2V_CACHE % target, 'rb'))

    return (train_x, train_t, train_labels,
            test_x, test_t, test_labels, test_id, test_text,
            word_index, embedding_matrix)


def prepare_amp_dataset():
    raw_data = json.load(open(DirConfig.AMP_RAW_DATA, 'r'))
    all_ids = set()
    all_text = set()
    for tweet in raw_data:
        is_retweet = tweet['retweet_flag']
        if is_retweet == 'Y':
            continue
        tid = tweet['id']
        text = tweet['tweet'].replace('\n', ' ').replace('\r', '').replace('&amp;', '')
        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)
        text = text.strip()
        if len(text) == 0:
            continue
        if len(text.split(' ')) <= 5:
            continue
        if tid in all_ids:
            continue
        all_ids.add(tid)
        all_text.add(text)
    all_text = list(all_text)
    num = len(all_text)
    print(num)
    with open(DirConfig.TEST_AMP_FILE, 'w') as f:
        f.write('ID\tTarget\tTweet\tStance\n')
        for i in range(num):
            f.write(str(i+1) + '\t' + 'AMP' + '\t' + all_text[i] + '\t' + 'NONE' + '\n')
    print('saved.')


if __name__ == '__main__':
    print('---- loading word2vec ...')
    word2vec = load_word_embedding(TrainConfig.W2V_TYPE, DirConfig.GLOVE_FILE)
    text2matrix('cc', 'cc', word2vec, 'windows-1252')
