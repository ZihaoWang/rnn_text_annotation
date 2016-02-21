#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from rnn import *

use_gpu(1)

import data
drop_rate = 0.
batch_size = 1
input_size = 9235
output_size = 104
#seqs, i2w, w2i, data_xy = data.char_sequence("/data/toy.txt", batch_size)
seqs, i2w, w2i, data_xy, existing_annos = data.char_sequence("/gds/zhwang/zhwang/data/cuhk/testing_data", batch_size, input_size, output_size)
hidden_size = [100, 100]
#dim_x = len(w2i)
dim_x = input_size
dim_y = output_size
print dim_x, dim_y

cell = "gru" # cell = "gru" or "lstm"
optimizer = "adadelta"

print "building..."
model = RNN(dim_x, dim_y, hidden_size, cell, optimizer, drop_rate)
model = load_model("./model/char_rnn.model", model) 

num_x = 0.0
acc = 0.0
idx = 0
for each_batch in data_xy:
    label = np.argmax(each_batch.y, axis = 1)
    activation = model.predict(each_batch.x, each_batch.mask, each_batch.local_batch_size)[0]
    p_label = np.argmax(activation, axis = 1)

    for c in xrange(len(label)):
        num_x += 1
        if label[c] == p_label[c]:
            acc += 1
    idx += 1
    print idx
print "Accuracy = " + str(acc / num_x)

'''
X = np.zeros((1, dim_x), np.float32)
a = "r"
X[0, w2i[a]] = 1
print a,
for i in xrange(100):
    Y = model.predict(X, np.ones((X.shape[0], 1), np.float32),  1)[0]
    Y = Y[Y.shape[0] - 1,:]
    p_label = np.argmax(Y)
    print i2w[p_label],
    X = np.concatenate((X, np.reshape(Y, (1, len(Y)))), axis=0)
'''
