#imports
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math


class biGRU(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False, emb_drop=0.6, pad_idx=0):
        super(biGRU, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.emb_drop = nn.Dropout(emb_drop)

        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))

        self.b = nn.Parameter(torch.FloatTensor([0]))

        self.init_params_()

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        c, r = self.forward_enc(x1, x2)
        o = self.forward_fc(c, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        _, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        #concatenate both layers
        c = torch.cat([c[0], c[1]], dim=-1)
        r = torch.cat([r[0], r[1]], dim=-1)

        return c.squeeze(), r.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        # (batch_size x 1 x h_dim)
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o


class A_DE_bigRU(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False, emb_drop=0.6, pad_idx=0):
        super(A_DE_bigRU, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.emb_drop = nn.Dropout(emb_drop)

        self.M = nn.Parameter(torch.FloatTensor(h_dim, h_dim))

        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(h_dim, h_dim)
        self.init_params_()

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        # (batch_size x 1 x h_dim)
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o


class biCGRU(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, gpu=False, emb_drop=0.6, pad_idx=0):
        super(biCGRU, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)

        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.emb_drop = nn.Dropout(emb_drop)

        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))

        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2 * h_dim, 2 * h_dim)
        self.init_params_()

        if gpu:
            self.cuda()

    def init_params_(self):
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        sc, c, r = self.forward_enc(x1, x2)
        c_attn = self.forward_attn(sc, r, x1mask)
        o = self.forward_fc(c_attn, r)

        return o.view(-1)

    def forward_enc(self, x1, x2):
        """
        x1, x2: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        x1_emb = self.emb_drop(self.word_embed(x1))
        x2_emb = self.emb_drop(self.word_embed(x2))

        # Each is (1 x batch_size x h_dim)
        sc, c = self.rnn(x1_emb)
        _, r = self.rnn(x2_emb)
        #concatenate both layers
        c = torch.cat([c[0], c[1]], dim=-1)
        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        c, r: tensor of (batch_size, h_dim)
        """
        # (batch_size x 1 x h_dim)
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o


class Add_GRU(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(Add_GRU, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        #Load pre-trained embedding
        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        #size of description RNN
        self.desc_rnn_size = 100

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.init_params_()
        self.tech_w = 0.0
        if gpu:
            self.cuda()

    def init_params_(self):
        #Initializing parameters
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask, x2mask, key_r, key_mask_r):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        #Masking for attention in
        key_mask_r = key_mask_r.unsqueeze(2).repeat(1, 1, self.desc_rnn_size * 2)
        key_emb_r = self.get_weighted_key(key_r, key_mask_r)
        #get all states from gru
        sc, sr, c, r = self.forward_enc(x1, x2, key_emb_r)
        #getting values after applying attention
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask)
        #final output
        o = self.forward_fc(c_attn, r_attn)

        return o.view(-1)

    def get_weighted_key(self, key_r, key_mask_r):
        """
        get the output from desc gru
        response_keys, response_mask: seqs of words (batch_size, seq_len)
        """
        #batch_size
        b_s = key_r.size(0)
        #sequence length
        s_len = key_r.size(1)
        key_emb = self.emb_drop(self.word_embed(key_r.view(b_s * s_len, -1)))
        key_emb = self._forward(key_emb)
        key_emb_r = key_emb.view(b_s, s_len, -1) * key_mask_r
        del (key_emb, b_s, s_len)

        return key_emb_r

    def _forward(self, x):
        """
        get description embeddings
        :param x:
        :return:
        """
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out.squeeze()

    def forward_enc(self, x1, x2, key_emb_r):
        """
        x1, x2, key_emb: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        b, s = x2.size(0), x2.size(1)
        x1_emb = self.emb_drop(self.word_embed(x1)) # B X S X E
        sc, c = self.rnn(x1_emb)
        c = torch.cat([c[0], c[1]], dim=-1)  # concat the bi-directional hidden layers, shape = B X H

        x2_emb = self.emb_drop(self.word_embed(x2))
        #adding the embeddings
        x2_emb = x2_emb + key_emb_r
        # Each is (1 x batch_size x h_dim)

        sr, r = self.rnn(x2_emb)

        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, sr, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        dual encoder
        c, r: tensor of (batch_size, h_dim)
        """
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o


class AK_DE_biGRU(nn.Module):

    def __init__(self, emb_dim, n_vocab, h_dim=256, pretrained_emb=None, pad_idx=0, gpu=False, emb_drop=0.6, max_seq_len=160):
        super(AK_DE_biGRU, self).__init__()

        self.word_embed = nn.Embedding(n_vocab, emb_dim, padding_idx=pad_idx)
        #Load pre-trained embedding
        if pretrained_emb is not None:
            self.word_embed.weight.data.copy_(pretrained_emb)
        #size of description RNN
        self.desc_rnn_size = 100

        self.rnn = nn.GRU(
            input_size=emb_dim, hidden_size=h_dim,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.rnn_desc = nn.GRU(
            input_size=emb_dim, hidden_size=self.desc_rnn_size,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.h_dim = h_dim
        self.emb_dim = emb_dim
        self.emb_drop = nn.Dropout(emb_drop)
        self.max_seq_len = max_seq_len
        self.M = nn.Parameter(torch.FloatTensor(2*h_dim, 2*h_dim))
        self.b = nn.Parameter(torch.FloatTensor([0]))
        self.Wc = nn.Parameter(torch.FloatTensor(2*h_dim, emb_dim))
        self.We = nn.Parameter(torch.FloatTensor(emb_dim, emb_dim))
        self.attn = nn.Linear(2*h_dim, 2*h_dim)
        self.init_params_()
        self.tech_w = 0.0
        if gpu:
            self.cuda()

    def init_params_(self):
        #Initializing parameters
        nn.init.xavier_normal(self.M)

        # Set forget gate bias to 2
        size = self.rnn.bias_hh_l0.size(0)
        self.rnn.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn.bias_ih_l0.size(0)
        self.rnn.bias_ih_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_hh_l0.size(0)
        self.rnn_desc.bias_hh_l0.data[size//4:size//2] = 2

        size = self.rnn_desc.bias_ih_l0.size(0)
        self.rnn_desc.bias_ih_l0.data[size//4:size//2] = 2

    def forward(self, x1, x2, x1mask, x2mask, key_r, key_mask_r):
        """
        Inputs:
        -------
        x1, x2: seqs of words (batch_size, seq_len)

        Outputs:
        --------
        o: vector of (batch_size)
        """
        #Masking for attention in
        key_mask_r = key_mask_r.unsqueeze(2).repeat(1, 1, self.desc_rnn_size * 2)
        key_emb_r = self.get_weighted_key(key_r, key_mask_r)
        #get all states from gru
        sc, sr, c, r = self.forward_enc(x1, x2, key_emb_r)
        #getting values after applying attention
        c_attn = self.forward_attn(sc, r, x1mask)
        r_attn = self.forward_attn(sr, c, x2mask)
        #final output
        o = self.forward_fc(c_attn, r_attn)

        return o.view(-1)

    def get_weighted_key(self, key_r, key_mask_r):
        """
        get the output from desc gru
        response_keys, response_mask: seqs of words (batch_size, seq_len)
        """
        #batch_size
        b_s = key_r.size(0)
        #sequence length
        s_len = key_r.size(1)
        key_emb = self.emb_drop(self.word_embed(key_r.view(b_s * s_len, -1)))
        key_emb = self._forward(key_emb)
        key_emb_r = key_emb.view(b_s, s_len, -1) * key_mask_r
        del (key_emb, b_s, s_len)

        return key_emb_r

    def _forward(self, x):
        """
        get description embeddings
        :param x:
        :return:
        """
        _, h = self.rnn_desc(x)
        out = torch.cat([h[0], h[1]], dim=-1)

        return out.squeeze()

    def forward_enc(self, x1, x2, key_emb_r):
        """
        x1, x2, key_emb: seqs of words (batch_size, seq_len)
        """
        # Both are (batch_size, seq_len, emb_dim)
        b, s = x2.size(0), x2.size(1)
        x1_emb = self.emb_drop(self.word_embed(x1)) # B X S X E
        sc, c = self.rnn(x1_emb)
        c = torch.cat([c[0], c[1]], dim=-1)  # concat the bi-directional hidden layers, shape = B X H

        c_k = c.unsqueeze(1).repeat(1, key_emb_r.size(1), 1)

        x2_emb = self.emb_drop(self.word_embed(x2))
        #Equation 10
        alpha_k = F.softmax(torch.mm(c_k.view(b*s, -1), self.Wc).view(b, s, self.emb_dim) + torch.mm(key_emb_r.view(b*s, -1), self.We).view(b, s, self.emb_dim), dim=-1)
        #Equation 11
        x2_emb = (1 - alpha_k) * x2_emb + alpha_k * key_emb_r
        # Each is (1 x batch_size x h_dim)

        sr, r = self.rnn(x2_emb)

        r = torch.cat([r[0], r[1]], dim=-1)

        return sc, sr, c.squeeze(), r.squeeze()

    def forward_attn(self, x1, x2, mask):
        """
        attention
        :param x1: batch X seq_len X dim
        :return:
        """
        max_len = x1.size(1)
        b_size = x1.size(0)

        x2 = x2.squeeze(0).unsqueeze(2)
        attn = self.attn(x1.contiguous().view(b_size*max_len, -1))# B, T,D -> B*T,D
        attn = attn.view(b_size, max_len, -1) # B,T,D
        attn_energies = (attn.bmm(x2).transpose(1, 2)) #B,T,D * B,D,1 --> B,1,T
        alpha = F.softmax(attn_energies.squeeze(1), dim=-1)  # B, T
        alpha = alpha * mask  # B, T
        alpha = alpha.unsqueeze(1)  # B,1,T
        weighted_attn = alpha.bmm(x1)  # B,T

        return weighted_attn.squeeze()

    def forward_fc(self, c, r):
        """
        dual encoder
        c, r: tensor of (batch_size, h_dim)
        """
        o = torch.mm(c, self.M).unsqueeze(1)
        # (batch_size x 1 x 1)
        o = torch.bmm(o, r.unsqueeze(2))
        o = o + self.b

        return o
