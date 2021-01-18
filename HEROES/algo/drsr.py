import tensorflow as tf
import numpy as np
import random
import sys
sys.path.append("../")
from utils.base import BaseModel
from utils.base import cal_acc, cal_auc, cal_f1, cal_logloss, cal_ndcg, cal_map

class DRSR(BaseModel):
    def __init__(self, sess, l2_weight, lr, grad_clip, 
                batch_size, tf_device, keep_prob, signal_weight, observation_weight,
                feature_dim, hidden_dim, position_dim, seq_len, feature_num,
                signal_type, data_name, is_binary, name="DRSR"):
        super(DRSR, self).__init__(sess, name)

        self.feature_dim = feature_dim  # feature dim for feature embedding
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.position_dim = position_dim  # feature dim for position
        self.l2_weight = l2_weight
        self.seq_len = seq_len
        self.data_name = data_name
        self.signal_type = signal_type  # ctr, cvr
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        self.signal_weight = signal_weight
        self.observation_weight = observation_weight
        self.is_binary = is_binary  # size of output

        with tf.device(tf_device):
            if self.data_name == "taobao":
                self.featureid_ph = tf.sparse_placeholder(tf.int32, (None, self.seq_len, None), name="featureid")
                self.featurevalue_ph = tf.sparse_placeholder(tf.float32, (None, self.seq_len, None), name="featurevalue")
            elif self.data_name == "criteo":
                self.featureid_ph = tf.placeholder(tf.float32, (None, self.seq_len, 2), name="featureid") 
                self.featurevalue_ph = tf.placeholder(tf.int32, (None, self.seq_len, 10), name="featurevalue") 
            else:
                raise NotImplementedError   
            self.observation_ph = tf.placeholder(tf.int32, (None, self.seq_len, 2), name="observation")
            self.click_ph = tf.placeholder(tf.int32, (None, self.seq_len, 2), name="click")
            self.conversion_ph = tf.placeholder(tf.int32, (None, self.seq_len, 2), name="conversion")

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
        # emb: batch_size, seq_len, position+feature dim
        emb = tf.concat([feature, position_emb], axis=-1)
        emb = tf.layers.dense(emb, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal())
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob, output_keep_prob=self.keep_prob)
        seq_len = [self.seq_len for _ in range(self.batch_size)]
        pred, _ = tf.nn.dynamic_rnn(cell, emb, dtype=tf.float32, time_major=False, sequence_length=seq_len)  # batch_size, seq_len, hidden_dim
        pred = tf.layers.dense(pred, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal())
        out_dim = 1 if self.is_binary else 2
        pred = tf.layers.dense(pred, units=out_dim, activation=tf.nn.relu)
        # censor_len: batch_size, seq_len, 1
        self.seq_mask = tf.reduce_sum(self.observation_ph, axis=-1, keepdims=True)
        # seq_mask = self.seq_mask if self.is_binary else tf.repeat(self.seq_mask, repeats=2, axis=-1)
        # self.seq_mask = tf.sequence_mask(censor_len, maxlen=out_dim, dtype=tf.float32)
        # pred_score: batch_size, seq_len, 2
        pred = tf.sigmoid(pred*seq_mask)
        return pred
    
    def build_network(self):
        feature_ph = self._load_feature()
        self.DRSR_net = tf.make_template("DRSR", self._network_template)
        self.DRSR_tf = self.DRSR_net(feature_ph)

    def _load_feature(self):
        if self.data_name == "taobao":
            features = tf.nn.embedding_lookup_sparse(self.feature_embeddingtable, sp_ids=self.featureid_ph, sp_weights=self.featurevalue_ph) # batch_size*seq_len, feature_dim
            features = tf.reshape(features, [-1, self.seq_len, self.feature_dim]) # batch_size, seq_len, 1*feature_dim
            feature_ph = tf.layers.dense(features, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal()) # # batch_size, seq_len, hidden_dim
        elif self.data_name == "criteo":
            features = tf.nn.embedding_lookup(self.feature_embeddingtable, self.featurevalue_ph) 
            features = tf.reshape(features, [-1, self.seq_len, 10*self.feature_dim]) # batch_size, seq_len, 10*feature_dim
            features = tf.concat((features, self.featureid_ph), axis=2) # batch_size, seq_len, 10*feature_dim+2
            feature_ph = tf.layers.dense(features, units=self.hidden_dim, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal()) # # batch_size, seq_len, hidden_dim
        else:
            raise NotImplementedError
        return feature_ph
    
    def build_train_op(self):
        print("===== BUG =====")
        if self.signal_type == "ctr":
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.click_ph[:,:,0], logits=self.DRSR_tf)
            self.loss = tf.reduce_sum(self.loss*self.seq_mask)/self.batch_size
        elif self.signal_type == "cvr":
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.conversion_ph[:,:,0], logits=self.DRSR_tf)
            self.loss = tf.reduce_sum(self.loss*self.seq_mask)/self.batch_size
        else:
            raise NameError
        trainable_vars = tf.trainable_variables()
        l2_loss = tf.add_n([tf.nn.l2_loss(_var) for _var in trainable_vars])
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss*self.signal_weight+l2_loss*self.l2_weight, trainable_vars), self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(grads, trainable_vars))
    
    def train(self, train_data, print_interval):
        total_num = len(train_data)
        batch_num = total_num//self.batch_size
        loss = []
        # metrics
        acc, auc, f1, logloss = [], [], [], []
        ndcg3, ndcg5, ndcg10, map3, map5, map10 = [], [], [], [], [], [] 
        for _iter in range(batch_num):
            batch_documentlist, batch_featurelist, batch_clicklist, batch_conversionlist, batch_observationlist = train_data.sample()
            train_feed_dict = {
                self.observation_ph: batch_observationlist,
                self.featureid_ph: batch_documentlist,
                self.featurevalue_ph: batch_featurelist,
                self.click_ph: batch_clicklist,
                self.conversion_ph: batch_conversionlist
            }
            batch_loss, batch_predictionlist, _ = self.sess.run([self.loss, self.DRSR_tf, self.train_op])
            loss.append(batch_loss)
            if self.signal_type == "ctr":
                batch_signallist = batch_clicklist
            elif self.signal_type == "cvr":
                batch_signallist = batch_conversionlist
            else:
                raise NameError
            acc.append(cal_acc(batch_predictionlist, batch_signallist, isbinary=False))
            auc.append(cal_auc(batch_predictionlist, batch_signallist, isbinary=False))
            f1.append(cal_f1(batch_predictionlist, batch_signallist, isbinary=False))
            logloss.append(cal_logloss(batch_predictionlist, batch_signallist, isbinary=False))
            ndcg3.append(cal_ndcg(3, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            ndcg5.append(cal_ndcg(5, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            ndcg10.append(cal_ndcg(10, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            map3.append(cal_map(3, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            map5.append(cal_map(5, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            map10.append(cal_map(10, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            if _iter % print_interval == 0:
                print("===== BATCH #{:<4f}, LOSS [{:<.6f}]").format(_iter, np.mean(loss))
        return (np.mean(acc), np.mean(auc), np.mean(f1), np.mean(logloss), np.mean(ndcg3), np.mean(ndcg5), np.mean(ndcg10), 
                np.mean(map3), np.mean(map5), np.mean(map10))
    
    def test(self, test_data, print_interval):
        total_num = len(test_data)
        batch_num = total_num//self.batch_size
        loss = []
        # metrics
        acc, auc, f1, logloss = [], [], [], []
        ndcg3, ndcg5, ndcg10, map3, map5, map10 = [], [], [], [], [], [] 
        for _iter in range(batch_num):
            batch_documentlist, batch_featurelist, batch_clicklist, batch_conversionlist, batch_observationlist = train_data.sample()
            train_feed_dict = {
                self.observation_ph: batch_observationlist,
                self.featureid_ph: batch_documentlist,
                self.featurevalue_ph: batch_featurelist,
                self.click_ph: batch_clicklist,
                self.conversion_ph: batch_conversionlist
            }
            batch_loss, batch_predictionlist, _ = self.sess.run([self.loss, self.DRSR_tf, self.train_op])
            loss.append(batch_loss)
            if self.signal_type == "ctr":
                batch_signallist = batch_clicklist
            elif self.signal_type == "cvr":
                batch_signallist = batch_conversionlist
            else:
                raise NameError
            acc.append(cal_acc(batch_predictionlist, batch_signallist, isbinary=self.isbinary))
            auc.append(cal_auc(batch_predictionlist, batch_signallist, isbinary=self.isbinary))
            f1.append(cal_f1(batch_predictionlist, batch_signallist, isbinary=self.isbinary))
            logloss.append(cal_logloss(batch_predictionlist, batch_signallist, isbinary=self.isbinary))
            ndcg3.append(cal_ndcg(3, batch_predictionlist, batch_signallist, self.seq_len, isbinary=self.isbinary))
            ndcg5.append(cal_ndcg(5, batch_predictionlist, batch_signallist, self.seq_len, isbinary=self.isbinary))
            ndcg10.append(cal_ndcg(10, batch_predictionlist, batch_signallist, self.seq_len, isbinary=self.isbinary))
            map3.append(cal_map(3, batch_predictionlist, batch_signallist, self.seq_len, isbinary=self.isbinary))
            map5.append(cal_map(5, batch_predictionlist, batch_signallist, self.seq_len, isbinary=self.isbinary))
            map10.append(cal_map(10, batch_predictionlist, batch_signallist, self.seq_len, isbinary=False))
            if _iter % print_interval == 0:
                print("===== BATCH #{:<4f}, LOSS [{:<.6f}]").format(_iter, np.mean(loss))
        return (np.mean(acc), np.mean(auc), np.mean(f1), np.mean(logloss), np.mean(ndcg3), np.mean(ndcg5), np.mean(ndcg10), 
                np.mean(map3), np.mean(map5), np.mean(map10))