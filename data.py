# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

class raw_data:
    def __init__(self):
        self.word = []
        self.anno = []


class batch_data:
    def __init__(self, local_topic, w2i, max_sent_len):
        dic_size = len(w2i)
        local_batch_size = len(local_topic)
        self.x = np.zeros((max_sent_len, dic_size * local_batch_size), dtype = theano.config.floatX)
        self.y = np.zeros((max_sent_len, local_batch_size), dtype = theano.config.floatX)
        self.mask = np.zeros((max_sent_len, local_batch_size), dtype = theano.config.floatX)
        self.batch_size = local_batch_size

        for idx_sent in xrange(self.batch_size):
            each_sent = local_topic[idx_sent]
            x_offset = idx_sent * dic_size
            for idx_word in xrange(len(each_sent.word)):
                self.x[idx_word, x_offset + w2i[each_sent.word[idx_word]]] = 1
                self.y[idx_word, idx_sent] = each_sent.anno[idx_word]
                self.mask[idx_word, idx_sent] = 1

def char_sequence(f_path = None, batch_size = 1):
    seqs = []
    i2w = {}
    w2i = {}
    topic = []
    if f_path == None:
        f = open(curr_path + "/data/toy.txt", "r")
    else:
        f = open(f_path, "r")

    each_sent = raw_data()
    max_sent_len = 0
    for line in f:
        line = line.strip('\n').lower()
        if line == "----":
            continue
        if line == "":
            if len(each_sent.word) > max_sent_len:
                max_sent_len = len(each_sent.word)
            topic.append(each_sent)
            each_sent = raw_data()
            continue

        pair = line.split()
        each_sent.word.append(pair[0])
        each_sent.anno.append(pair[1])
        if pair[0] not in w2i:
            i2w[len(w2i)] = pair[0]
            w2i[pair[0]] = len(w2i)

    f.close()

    data_xy = []
    num_batch = len(topic) / batch_size + 1
    for i in xrange(num_batch - 1):
        each_batch = batch_data(topic[i * batch_size : (i + 1) * batch_size], w2i, max_sent_len)
        data_xy.append(each_batch)
    last_batch = batch_data(topic[(num_batch - 1) * batch_size : ], w2i, max_sent_len)
    data_xy.append(last_batch)

    print "#dic = " + str(len(w2i))
    return topic, i2w, w2i, data_xy

def batch_sequences(topic, dim, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(topic_bags)):
        seq = topic_bags[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

        if len(batch_x) == batch_size or (i == len(topic_bags) - 1):
            max_len = np.max(seqs_len);
            mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
            concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)
            concat_Y = concat_X.copy()

            for b_i in xrange(len(batch_x)):
                X = batch_x[b_i]
                Y = batch_y[b_i]
                mask[0 : X.shape[0], b_i] = 1
                for r in xrange(max_len - X.shape[0]):
                    X = np.concatenate((X, zeros_m), axis=0)
                    Y = np.concatenate((Y, zeros_m), axis=0)
                concat_X[:, b_i * dim : (b_i + 1) * dim] = X 
                concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y
            data_xy[batch_id] = [concat_X, concat_Y, mask, len(batch_x)]
            batch_x = []
            batch_y = []
            seqs_len = []
            batch_id += 1
    return data_xy

def load_hlm(f_path, batch_size = 1):
    jieba.load_userdict("./data/hlm/name.dic")
    seqs = []
    i2w = {}
    w2i = {}
    lines = []
    data_xy = {}
    f = open(curr_path + "/" + f_path, "r")
    for line in f:
        line = line.strip('\n').lower()
        if len(line) < 3 or "手机电子书" in line:
            continue
        seg_list = jieba.cut(line)

        w_line = []
        for w in seg_list:
            if w not in w2i:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            w_line.append(w)
            if len(w_line) == 100:
                lines.append(w_line)
                w_line = []
        if len(w_line) < 100:
            lines.append(w_line)
    f.close
    seqs = lines
    data_xy = batch_index(seqs, i2w, w2i, batch_size)
    print "#dic = " + str(len(w2i))
    return seqs, i2w, w2i, data_xy

# limit memory
def batch_index(seqs, i2w, w2i, batch_size):
    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    for i in xrange(len(seqs)):
        batch_x.append(i)
        batch_y.append(i)
        if len(batch_x) == batch_size or (i == len(seqs) - 1):
            data_xy[batch_id] = [batch_x, batch_y, [], len(batch_x)]
            batch_x = []
            batch_y = []
            batch_id += 1
    return data_xy

def index2seqs(lines, x_index, w2i):
    seqs = []
    for i in x_index:
        line = lines[i]
        x = np.zeros((len(line), len(w2i)), dtype = theano.config.floatX)
        for j in range(0, len(line)):
            x[j, w2i[line[j]]] = 1
        seqs.append(np.asmatrix(x))

    data_xy = {}
    batch_x = []
    batch_y = []
    seqs_len = []
    batch_id = 0
    dim = len(w2i)
    zeros_m = np.zeros((1, dim), dtype = theano.config.floatX)
    for i in xrange(len(seqs)):
        seq = seqs[i];
        X = seq[0 : len(seq) - 1, ]
        Y = seq[1 : len(seq), ]
        batch_x.append(X)
        seqs_len.append(X.shape[0])
        batch_y.append(Y)

    max_len = np.max(seqs_len);
    mask = np.zeros((max_len, len(batch_x)), dtype = theano.config.floatX)
    concat_X = np.zeros((max_len, len(batch_x) * dim), dtype = theano.config.floatX)
    concat_Y = concat_X.copy()
    
    for b_i in xrange(len(batch_x)):
        X = batch_x[b_i]
        Y = batch_y[b_i]
        mask[0 : X.shape[0], b_i] = 1
        for r in xrange(max_len - X.shape[0]):
            X = np.concatenate((X, zeros_m), axis=0)
            Y = np.concatenate((Y, zeros_m), axis=0)
        concat_X[:, b_i * dim : (b_i + 1) * dim] = X 
        concat_Y[:, b_i * dim : (b_i + 1) * dim] = Y
    return concat_X, concat_Y, mask, len(batch_x)
