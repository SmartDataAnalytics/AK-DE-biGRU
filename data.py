from torchtext import data
from torchtext.vocab import Vocab, GloVe
import torch
from torch.autograd import Variable
import re
from collections import OrderedDict, Counter
import numpy as np
import pickle

URL_TOK = '__url__'
PATH_TOK = '__path__'


class UDCv1:
    """
    Wrapper for UDCv2 taken from: http://dataset.cs.mcgill.ca/ubuntu-corpus-1.0/.
    Everything has been preprocessed and converted to numerical indexes.
    """

    def __init__(self, path, batch_size=256, max_seq_len=160, use_mask=False, gpu=True, use_fasttext=False):
        self.batch_size = batch_size
        self.max_seq_len_c = max_seq_len
        self.max_seq_len_r = int(max_seq_len/2)
        self.use_mask = use_mask
        self.gpu = gpu

        self.desc_len = 44
        #load the dataset pickle file
        with open(f'{path}/dataset_1M.pkl', 'rb') as f:
            dataset = pickle.load(f, encoding='ISO-8859-1')
            self.train, self.valid, self.test = dataset
        #load the fasttext vector
        if use_fasttext:
            vectors = np.load(f'{path}/fast_text_200_v.npy')
            #vectors = np.load(f'{path}/w2vec_200.npy')
            #man_vec = np.load(f'{path}/key_vec.npy')
        else:
            with open(f'{path}/W.pkl', 'rb') as f:
                vectors, _ = pickle.load(f, encoding='ISO-8859-1')
        #load the command description file
        self.ubuntu_cmd_vec = np.load(f'{path}/command_description.npy').item()
        #self.ubuntu_cmd_vec = np.load(f'{path}/man_dict_key.npy').item()

        print('Finished loading dataset!')

        self.n_train = len(self.train['y'])
        self.n_valid = len(self.valid['y'])
        self.n_test = len(self.test['y'])
        self.vectors = torch.from_numpy(vectors.astype(np.float32))
        #self.man_vec = torch.from_numpy(man_vec.astype(np.float32))

        self.vocab_size = self.vectors.size(0)
        self.emb_dim = self.vectors.size(1)

    def get_iter(self, dataset='train'):
        if dataset == 'train':
            dataset = self.train
        elif dataset == 'valid':
            dataset = self.valid
        else:
            dataset = self.test

        for i in range(0, len(dataset['y']), self.batch_size):
            c = dataset['c'][i:i+self.batch_size]
            r = dataset['r'][i:i+self.batch_size]
            y = dataset['y'][i:i+self.batch_size]


            c, r, y, c_mask, r_mask, key_r, key_mask_r = self._load_batch(c, r, y, self.batch_size)

            if self.use_mask:
                yield c, r, y, c_mask, r_mask, key_r, key_mask_r
            else:
                yield c, r, y


    def get_key(self, sentence, max_seq_len, max_len):
        """
        get key mask
        :param sentence:
        :param max_len:
        :return:
        """
        key_mask = np.zeros((max_seq_len))
        keys = np.zeros((max_seq_len, max_len))
        for j, word in enumerate(sentence):
            if int(word) in self.ubuntu_cmd_vec.keys():
                keys[j] = self.ubuntu_cmd_vec[int(word)][:max_len]
                key_mask[j] = 1
            else:
                keys[j] = np.zeros((max_len))
        return key_mask, keys


    def _load_batch(self, c, r, y, size):
        c_arr = np.zeros([size, self.max_seq_len_c], np.int)
        r_arr = np.zeros([size, self.max_seq_len_r], np.int)
        y_arr = np.zeros(size, np.float32)

        c_mask = np.zeros([size, self.max_seq_len_c], np.float32)
        r_mask = np.zeros([size, self.max_seq_len_r], np.float32)

        #key_c = np.zeros([size, self.max_seq_len_c, self.desc_len], np.float32)
        key_r = np.zeros([size, self.max_seq_len_r, self.desc_len], np.float32)

        #key_mask_c = np.zeros([size, self.max_seq_len_c], np.float32)
        key_mask_r = np.zeros([size, self.max_seq_len_r], np.float32)

        for j, (row_c, row_r, row_y) in enumerate(zip(c, r, y)):

            # Truncate
            row_c = row_c[:self.max_seq_len_c]
            row_r = row_r[:self.max_seq_len_r]

            c_arr[j, :len(row_c)] = row_c
            r_arr[j, :len(row_r)] = row_r
            y_arr[j] = float(row_y)


            c_mask[j, :len(row_c)] = 1
            r_mask[j, :len(row_r)] = 1

            #key_mask_c[j], key_c[j] = self.get_key(row_c, self.max_seq_len_c, self.desc_len)
            key_mask_r[j], key_r[j] = self.get_key(row_r, self.max_seq_len_r, self.desc_len)

        # Convert to PyTorch tensor
        c = Variable(torch.from_numpy(c_arr))
        r = Variable(torch.from_numpy(r_arr))
        y = Variable(torch.from_numpy(y_arr))
        c_mask = Variable(torch.from_numpy(c_mask))
        r_mask = Variable(torch.from_numpy(r_mask))


        #key_mask_c = Variable(torch.from_numpy(key_mask_c), requires_grad = False)
        key_mask_r = Variable(torch.from_numpy(key_mask_r), requires_grad = False)

        #key_c = Variable(torch.from_numpy(key_c)).type(torch.LongTensor)
        key_r = Variable(torch.from_numpy(key_r)).type(torch.LongTensor)



        # Load to GPU
        if self.gpu:
            c, r, y = c.cuda(), r.cuda(), y.cuda()
            c_mask, r_mask = c_mask.cuda(), r_mask.cuda()
            #r_mask = r_mask.cuda()
            #key_c, key_mask_c, key_r, key_mask_r = key_c.cuda(), key_mask_c.cuda(), key_r.cuda(), key_mask_r.cuda()
            key_r, key_mask_r = key_r.cuda(), key_mask_r.cuda()

        return c, r, y, c_mask, r_mask, key_r, key_mask_r
