from __future__ import division
import argparse
import cPickle
import lasagne
import lasagne as nn
import numpy as np
import pyprind
import re
import sys
import theano
import theano.tensor as T
import time
from blocks import initialization
from blocks.bricks import Linear, Tanh
from blocks.bricks.recurrent import GatedRecurrent, LSTM, SimpleRecurrent
from collections import defaultdict, OrderedDict
from theano.ifelse import ifelse
from theano.printing import Print as pp
from lasagne import nonlinearities, init, utils
from lasagne.layers import Layer, InputLayer, DenseLayer, helper
sys.setrecursionlimit(10000)
np.random.seed(42)

def initialize(L, W, b):
    for bricks in L:
        bricks.weights_init = W
        bricks.biases_init = b
        bricks.initialize()

class Model(object):
    def __init__(self,
                 data,
                 U,
                 img_h=160,
                 img_w=300,
                 hidden_size=100,
                 batch_size=50,
                 lr=0.001,
                 lr_decay=0.95,
                 fine_tune_W=True,
                 fine_tune_M=False,
                 optimizer='adam',
                 forget_gate_bias=2,
                 encoder='rnn'):
        self.data = data
        self.img_h = img_h
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer

        index = T.iscalar()
        c = T.imatrix('c')
        r = T.imatrix('r')
        y = T.ivector('y')
        c_mask = T.fmatrix('c_mask')
        r_mask = T.fmatrix('r_mask')
        c_seqlen = T.ivector('c_seqlen')
        r_seqlen = T.ivector('r_seqlen')
        embeddings = theano.shared(U, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(img_w, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)

        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], r.shape[1], embeddings.shape[1]))

        l_linear = Linear(input_dim=U.shape[1], output_dim=hidden_size)
        if encoder.find('lstm') > -1:
            l_recurrent = LSTM(dim=hidden_size, activation=Tanh())
        else:
            l_recurrent = SimpleRecurrent(dim=hidden_size, activation=Tanh())

        initialize([l_linear], W=initialization.Orthogonal(), b=initialization.Constant(0))
        initialize([l_recurrent], W=initialization.Orthogonal(), b=initialization.Constant(0))

        h_context = l_recurrent.apply(l_linear.apply(c_input), mask=c_mask)
        h_response = l_recurrent.apply(l_linear.apply(r_input), mask=r_mask)
        if encoder.find('lstm') > -1:
            h_context = h_context[0]
            h_response = h_response[0]

        e_context = h_context[T.arange(batch_size), c_seqlen].reshape((c.shape[0], hidden_size))
        e_response = h_response[T.arange(batch_size), r_seqlen].reshape((r.shape[0], hidden_size))

        dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))
        #dp = pp('dp')(dp)
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c', 'r']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, img_h), dtype=np.int32))
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, img_h), dtype=theano.config.floatX))
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32))

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()
        self.l_linear = l_linear
        self.l_recurrent = l_recurrent
        self.embeddings = embeddings
        self.c = c
        self.r = r
        self.y = y
        self.c_mask = c_mask
        self.r_mask = r_mask
        self.c_seqlen = c_seqlen
        self.r_seqlen = r_seqlen

        self.update_params()

    def update_params(self):
        params = list(self.l_linear.parameters) + list(self.l_recurrent.parameters)[:-1]
        print params
        for p in params:
            print p.get_value().shape

        if self.fine_tune_W:
            params += [self.embeddings]
        if self.fine_tune_M:
            params += [self.M]

        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params

        if 'adam' == self.optimizer:
            updates = lasagne.updates.adam(self.cost, params, learning_rate=self.lr)
        elif 'adadelta' == self.optimizer:
            updates = lasagne.updates.adadelta(self.cost, params, learning_rate=1.0, rho=self.lr_decay)
        else:
            raise 'Unsupported optimizer: %s' % self.optimizer

        givens = {
            self.c: self.shared_data['c'],
            self.r: self.shared_data['r'],
            self.y: self.shared_data['y'],
            self.c_seqlen: self.shared_data['c_seqlen'],
            self.r_seqlen: self.shared_data['r_seqlen'],
            self.c_mask: self.shared_data['c_mask'],
            self.r_mask: self.shared_data['r_mask']
        }
        self.train_model = theano.function([], self.cost, updates=updates, givens=givens, on_unused_input='warn')
        self.get_loss = theano.function([], self.errors, givens=givens, on_unused_input='warn')
        self.get_probas = theano.function([], self.probas, givens=givens, on_unused_input='warn')

    def get_batch(self, dataset, index, max_l):
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        mask = np.zeros((self.batch_size,max_l), dtype=theano.config.floatX)
        batch = np.zeros((self.batch_size, max_l), dtype=np.int32)
        data = dataset[index*self.batch_size:(index+1)*self.batch_size]
        for i,row in enumerate(data):
            row = row[:max_l]
            batch[i,0:len(row)] = row
            seqlen[i] = len(row)-1
            mask[i,0:len(row)] = 1
        return batch, seqlen, mask

    def set_shared_variables(self, dataset, index):
        c, c_seqlen, c_mask = self.get_batch(dataset['c'], index, self.img_h)
        r, r_seqlen, r_mask = self.get_batch(dataset['r'], index, self.img_h)
        y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
        self.shared_data['c'].set_value(c)
        self.shared_data['r'].set_value(r)
        self.shared_data['y'].set_value(y)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        self.shared_data['r_mask'].set_value(r_mask)

    def compute_loss(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_loss()

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()[:,1]

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        best_val_perf = 0
        best_val_rk1 = 0
        test_perf = 0
        cost_epoch = 0

        n_train_batches = len(self.data['train']['y']) // self.batch_size
        n_val_batches = len(self.data['val']['y']) // self.batch_size
        n_test_batches = len(self.data['test']['y']) // self.batch_size

        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)
            if shuffle_batch:
                indices = np.random.permutation(indices)
            bar = pyprind.ProgBar(len(indices), monitor=True)
            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                cost_epoch = self.train_model()
                total_cost += cost_epoch
                self.set_zero(self.zero_vec)
                bar.update()
            end_time = time.time()
            print "cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)
            train_losses = [self.compute_loss(self.data['train'], i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / len(self.data['train']['y'])
            val_losses = [self.compute_loss(self.data['val'], i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / len(self.data['val']['y'])
            print 'epoch %i, train_perf %f, val_perf %f' % (epoch, train_perf*100, val_perf*100)

            val_probas = np.concatenate([self.compute_probas(self.data['val'], i) for i in xrange(n_val_batches)])
            val_recall_k = self.compute_recall_ks(val_probas)

            if val_perf > best_val_perf or val_recall_k[10][1] > best_val_rk1:
                best_val_perf = val_perf
                best_val_rk1 = val_recall_k[10][1]
                test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
                test_perf = 1 - np.sum(test_losses) / len(self.data['test']['y'])
                print 'test_perf: %f' % (test_perf*100)
                test_probas = np.concatenate([self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)])
                self.compute_recall_ks(test_probas)
            else:
                if not self.fine_tune_W:
                    self.fine_tune_W = True # try fine-tuning W
                else:
                    if not self.fine_tune_M:
                        self.fine_tune_M = True # try fine-tuning M
                    else:
                        break
                self.update_params()
        return test_perf

    def compute_recall_ks(self, probas):
      recall_k = {}
      for group_size in [2, 10]:
          recall_k[group_size] = {}
          print 'group_size: %d' % group_size
          for k in [1, 2, 5]:
              if k < group_size:
                  recall_k[group_size][k] = self.recall(probas, k, group_size)
                  print 'recall@%d' % k, recall_k[group_size][k]
      return recall_k

    def recall(self, probas, k, group_size):
        test_size = 10
        n_batches = len(probas) // test_size
        n_correct = 0
        for i in xrange(n_batches):
            batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
            #p = np.random.permutation(len(batch))
            #indices = p[np.argpartition(batch[p], -k)[-k:]]
            indices = np.argpartition(batch, -k)[-k:]
            if 0 in indices:
                n_correct += 1
        return n_correct / (len(probas) / test_size)

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)

def sgd_updates_adadelta(cost, params, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='embeddings'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != word_vec_name):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def pad_to_batch_size(X, batch_size):
    n_seqs = len(X)
    n_batches_out = np.ceil(float(n_seqs) / batch_size)
    n_seqs_out = batch_size * n_batches_out

    to_pad = n_seqs % batch_size
    if to_pad > 0:
        X += X[:batch_size-to_pad]
    return X

def get_nrows(fname):
    with open(fname, 'rb') as f:
        nrows = 0
        for _ in f:
            nrows += 1
        return nrows

def load_pv_vecs(fname, ndims):
    nrows = get_nrows(fname)
    with open(fname, "rb") as f:
        X = np.zeros((nrows+1, ndims))
        for i,line in enumerate(f):
            L = line.strip().split()
            X[i+1] = np.array(L[1:], dtype='float32')
        return X

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def main():
  parser = argparse.ArgumentParser()
  parser.register('type','bool',str2bool)
  parser.add_argument('--encoder', type=str, default='rnn', help='Encoder')
  parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
  parser.add_argument('--fine_tune_W', type='bool', default=False, help='Whether to fine-tune W')
  parser.add_argument('--fine_tune_M', type='bool', default=False, help='Whether to fine-tune M')
  parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
  parser.add_argument('--shuffle_batch', type='bool', default=False, help='Shuffle batch')
  parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
  parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
  parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
  parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
  parser.add_argument('--forget_gate_bias', type=float, default=2.0, help='Forget gate bias')
  parser.add_argument('--max_seqlen', type=int, default=160, help='Max seqlen')
  parser.add_argument('--input_dir', type=str, default='.', help='Input dir')
  parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
  parser.add_argument('--model_fname', type=str, default='model.pkl', help='Model filename')
  parser.add_argument('--dataset_fname', type=str, default='dataset.pkl', help='Dataset filename')
  parser.add_argument('--W_fname', type=str, default='W.pkl', help='W filename')
  args = parser.parse_args()
  print "args: ", args

  print "loading data...",
  train_data, val_data, test_data = cPickle.load(open('%s/%s' % (args.input_dir, args.dataset_fname), 'rb'))
  W, _ = cPickle.load(open('%s/%s' % (args.input_dir, args.W_fname), 'rb'))
  print "data loaded!"

  data = { 'train': train_data, 'val': val_data, 'test': test_data }

  model = Model(data,
                W.astype(theano.config.floatX),
                img_h=args.max_seqlen,
                img_w=W.shape[1],
                hidden_size=args.hidden_size,
                batch_size=args.batch_size,
                lr=args.lr,
                lr_decay=args.lr_decay,
                fine_tune_W=args.fine_tune_W,
                fine_tune_M=args.fine_tune_M,
                optimizer=args.optimizer,
                forget_gate_bias=args.forget_gate_bias,
                encoder=args.encoder)

  print model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)
  if args.save_model:
      cPickle.dump(model, open(args.model_fname, 'wb'))

if __name__ == '__main__':
  main()
