import re
import pickle
from collections import defaultdict

def getw2id(word, w2id):
    """
    get Ids of words from dictionary
    :param word:
    :param w2id:
    :return:
    """
    try:
        return w2id[word]
    except KeyError:
        return w2id['**unknown**']

def get_values(file, get_c_d=False, w2id=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    chars = []
    y = [int(a[0]) for a in data]
    c = [' __EOS__ '.join(a[1:-1]).split() for a in data]
    c = [[getw2id(w, w2id) for w in s] for s in c]
    r = [a[-1].split() for a in data]
    r = [[getw2id(w, w2id) for w in s] for s in r]
    if get_c_d:
        for word in c:
            sent = ' '.join(word)
            for char in sent:
                chars.append(char)
        chars = set(chars)
        return y, c, r, dict(zip(chars, range(len(chars))))
    else:
        return y, c, r


if __name__ == '__main__':
    #load the vocab file
    vocab = open('ubuntu_data/vocab.txt', 'r').readlines()
    w2id = {}
    for word in vocab:
        w = word.split('\n')[0].split('\t')
        w2id[w[0]] =int(w[1])

    train, test, valid = {}, {}, {}
    train['y'], train['c'], train['r'] = get_values('ubuntu_data/train.txt', get_c_d=False, w2id=w2id)
    test['y'], test['c'], test['r'] = get_values('ubuntu_data/test.txt', w2id=w2id)
    valid['y'], valid['c'], valid['r'] = get_values('ubuntu_data/valid.txt', w2id=w2id)
    #char_vocab = defaultdict(float)
    dataset = train, valid, test
    pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))