import tensorflow as tf
import numpy as np
import sys
import random
sys.path.append("../")
from algo.cell import Cell, MutliCell, CellState
from utils.base import BaseModel

# TIPS for improving the performance:
# 1. Add the batch normalization after MLP
# 2. Change the position embedding with tf.sin()
# 3. Add dropout after MLP
# 4. Data is orderly (not randomly) selected from Buffer

class HEROES(BaseModel):
    def __init__(self, sess, l2_weight, lr, grad_clip,
                name, batch_size, tf_device, ctr_weight, cvr_weight, boundary_weight, hidden_size,
                hidden_dim, position_dim, feature_dim, seq_len, feature_num,
                data_name, reuse):
        super(HEROES, self).__init__(sess, name)
    
        self.feature_dim = feature_dim # feature dim for embedding table
        self.hidden_dim = hidden_dim  # hidden dim for MLP layer
        if type(hidden_size) is list and len(hidden_size) == 2: # hidden size for RNN layer
            self.hidden_size = hidden_size
        elif type(hidden_size) is int:
            self.hidden_size = [hidden_size] * 2
        self.grad_clip = grad_clip
        self.position_dim = position_dim
        self.reuse = reuse
        self.data_name = data_name
        self.l2_weight = l2_weight
        self.lr = lr
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
        self.boundary_weight = boundary_weight

        with tf.device(tf_device):

            if self.data_name == "taobao":
                self.featureid_ph = tf.sparse_placeholder(tf.int32, (None, self.seq_len, None), name="featureid")
                self.featurevalue_ph = tf.sparse_placeholder(tf.float32, (None, self.seq_len, None), name="featurevalue")
            elif self.data_name == "criteo":
                self.featureid_ph = tf.placeholder(tf.float32, (None, self.seq_len, None), name="featureid") # None=2
                self.featurevalue_ph = tf.placeholder(tf.int32, (None, self.seq_len, None), name="featurevalue") # None=10
            else:
                raise NotImplementedError   
            self.observation_ph = tf.placeholder(tf.float32, (None, self.seq_len, 2), name="observation")
            self.click_ph = tf.placeholder(tf.float32, (None, self.seq_len, 2), name="click")
            self.conversion_ph = tf.placeholder(tf.float32, (None, self.seq_len, 2), name="conversion")

            # feature and position embedding 
            self.feature_embeddingmatrix = tf.random_normal(shape=[feature_num, self.feature_dim], stddev=0.1)
            self.feature_embeddingtable = tf.Variable(self.feature_embeddingmatrix)
            self.position_embeddingmatrix = tf.random_normal(shape=[self.seq_len, self.position_dim], stddev=0.1)
            self.position_embeddingtable = tf.Variable(self.position_embeddingmatrix)

            self.build_network()
            self.build_train_op()
            self.sess.run(tf.global_variables_initializer())
    
    def _network_template(self, feature):
        position = tf.tile(tf.range(self.seq_len), multiples=[self.batch_size])
        position = tf.reshape(position, [self.batch_size, self.seq_len])
        # position_emb: batch_size, seq_len, position_dim
        position_emb = tf.nn.embedding_lookup(self.position_embeddingmatrix, position)
        embed = tf.concat([feature, position_emb], axis=-1)
        embed = tf.layers.dense(embed, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
        # one for CTR and the other for CVR
        CTRcell = Cell(self.hidden_size[0], self.hidden_dim, self.hidden_size[1], self.reuse) # num_unit, h_below_size, h_above_size
        CVRcell = Cell(self.hidden_size[1], self.hidden_size[0], self.hidden_size[0], self.reuse)
        self.cell = MutliCell([CTRcell, CVRcell])
        # c, h: hidden_dim, z:1
        hidden_len = sum(self.hidden_size)*2+2
        # states: seq_len, batch_size, hidden_len
        states = tf.scan(self._build_recurrent, embed, initializer=tf.zeros([self.batch_size, hidden_len]))  # initializer for accum
        states = tf.transpose(states, [1, 2, 0]) # batch_size, seq_len, hidden_len
        states = self._split_cellstates(states) # batch_size, seq_len, 2
        self.indicator = tf.concat([s.z for s in states], axis=-1) # batch_size, seq_len, 2
        _predictor = []
        for _states in states: # batch_size, seq_len, 1
            _prediction = tf.concat([s.h for s in _states], axis=-1) # batch_size, seq_len*hidden_dim
            _prediction = tf.layers.dense(_prediction, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
            _predictor.append(_prediction)
        # censor_len: # batch_size, seq_len, 1
        censor_len = tf.reduce_sum(self.observation_ph, axis=-1, keep_dim=True)
        self.seq_mask = tf.sequence_mask(censor, max_len=self.seq_len, dtype=tf.float32)
        # use 2 for [1, 0] and [0, 1], also can use 1 for [1] and [0], try softmax
        CTRpredictor = tf.layers.dense(_predictor[0], units=2, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
        CTRpredictor = tf.sigmoid(CTRpredictor*self.seq_mask)
        CVRpredictor = tf.layers.dense(_predictor[1], units=2, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
        CVRpredictor = tf.sigmoid(CVRpredictor*self.seq_mask)
        return CTRpredictor, CVRpredictor
    
    def _split_cellstates(self, states):
        _split = []
        for _size in self.hidden_size:
            _split += [_size, _size, 1]
        _splitstates = tf.split(value=states, num_or_size_splits=_split, axis=-1)
        _cellstates = []
        for _l in range(2): # first for CTR, second for CVR
            c = _splitstates[l*3]    # batch_size, hidden_size
            h = _splitstates[l*3+1]  # batch_size, hidden_size
            z = _splitstates[l*3+2]  # batch_size, hidden_size
            _cellstates.append(CellState(c=c, h=h, z=z))
        return _cellstates
    
    def _gate_hiddenstates(self, hiddens):
        # hiddens: batch_size, sum(hiddem_size)
        # gate_hiddens: batch_size, sum(hidden_size)
        _gates, _gatehiddens = [], []
        for l in range(2): # for each layer   
            _hiddens = tf.layers.dense(hiddens, units=self.hidden_size[l], activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
            _gates.append(tf.sigmoid(_hiddens))
        _splits = tf.split(value=hiddens, num_or_size_splits=self.hidden_size, axis=-1)
        for _gate, _split in zip(_gates, _splits):
            _gatehiddens.append(tf.multiply(_gate, _split))
        return tf.concat(_gatehiddens, axis=1)

    def _build_recurrent(self, states, hiddens):
        # states: accum, hiddens: elem
        _cellstates = self._split_cellstates(states)  # batch_size, hidden_len -> (batch_size, hidden_size. batch_size, hidden_size. batch_size, 1)
        _habove = tf.concat([_cellstate.h for _cellstate in _cellstates], axis=1)  # batch_size, sum(hidden_size)
        _hiddens = tf.concat([hiddens, _habove], axis=1)  # batch_size, sum(hidden_size)+I
        _, _states = self.cell(_hiddens, _splitstates)  # (batch_size, hidden_dim. batch_size, hidden_dim. batch_size, 1)
        _states = [tf.concat(tuple(s), axis=1) for s in _states]  
        return tf.concat(_states, axis=1)  # batch_size, hidden_len
         
    def build_network(self):
        feature_ph = self._load_dataset()
        self.HEROES_net = tf.make_template("HEROES", self._network_template)
        self.CTR_tf, self.CVR_tf = self.HEROES_net(feature_ph)

    def _load_dataset(self):
        if self.data_name == "taobao":
            features = tf.nn.embedding_lookup_sparse(self.feature_embeddingtable, sp_ids=self.featureid_ph, sp_weights=self.featurevalue_ph) # batch_size*seq_len, feature_dim
            features = tf.reshape(features, [-1, self.seq_len, self.feature_dim]) # batch_size, seq_len, 1*feature_dim
            feature_ph = tf.layers.dense(features, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal()) # # batch_size, seq_len, hidden_dim
        elif self.data_name == "criteo":
            features = tf.nn.embedding_lookup(self.feature_embeddingtable, self.featurevalue_ph) 
            features = tf.reshape(features, [-1, self.seq_len, 10*self.feature_dim]) # batch_size, seq_len, 10*feature_dim
            features = tf.cat((features, self.featureid_ph), axis=2) # batch_size, seq_len, 10*feature_dim+2
            feature_ph = tf.layers.dense(features, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal()) # # batch_size, seq_len, hidden_dim
        else:
            raise NotImplementedError
        return feature_ph
    
    
    def build_train_op(self):
        self.CTR_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.click_ph, logits=self.CTR_tf)
        self.CTR_loss = tf.reduce
        self.CVR_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.click_ph, logits=self.CVR_tf)
        self.Boundary_loss = tf.nn.sparse_softmax
        trainable_vars = tf.trainable_variables()
        loss = 
        l2_loss = tf.add_n([tf.nn.l2_loss(_var) for _var in trainable_vars])*self.l2_weight
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss+l2_loss, trainable_vars), self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(grads, trainable_vars))

    def train(self, train_data, print_interval):
        total_num = len(train_data)
        batch_num = total_num//self.batch_size
        loss_record = [0.0, 0.0, 0.0] # CTR loss, CVR loss, censor loss
        for _iter in range(batch_num):
            train_feed_dict = {
                self.
            }
        