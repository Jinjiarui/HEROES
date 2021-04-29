import tensorflow as tf

import back_model


class Model(object):
    def __init__(self, placeholders, embedding_size, seq_max_len, max_features, n_hidden, n_classes, keep_prob,
                 prediction_embed_list, decay_step, lr, click_weight, conversion_weight, ctr_task_wgt, l2_reg,
                 position_embed=True, time_stamp=True, model_name='Heroes', dataset_name='Criteo'):
        self.placeholders = placeholders
        self.seq_max_len = seq_max_len
        self.ctr_task_wgt = ctr_task_wgt
        self.l2_reg = l2_reg
        self.lr = lr
        self.embedding_size = embedding_size
        self.position_embed = position_embed
        self.time_stamp = time_stamp
        self.position_embedding = tf.Variable(tf.random_normal([seq_max_len, embedding_size], stddev=0.1))
        self.embedding_matrix = tf.Variable(tf.random_normal([max_features, embedding_size], stddev=0.1))
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.model = self.get_back_model(seq_max_len, n_hidden, n_classes, keep_prob,
                                         prediction_embed_list, model_name)

        epsilon = 1e-7
        self.click_loss = lambda logits, labels: \
            -(1 - click_weight) / click_weight * labels * tf.log(logits + epsilon) - \
            (1 - labels) * tf.log(1 - logits + epsilon)
        self.conversion_loss = lambda logits, labels: \
            -(1 - conversion_weight) / conversion_weight * labels * tf.log(logits + epsilon) - \
            (1 - labels) * tf.log(1 - logits + epsilon)
        self.global_step = tf.Variable(0, trainable=False)
        cov_learning_rate = tf.train.exponential_decay(self.lr, self.global_step, decay_step, 0.7)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)

    def get_back_model(self, seq_max_len, n_hidden, n_classes, keep_prob,
                       prediction_embed_list, model_name='Heroes'):
        rnn_model_list = ['Heroes', 'motivate', 'motivate-single', 'RRN', 'time_LSTM', 'Motivate-Heroes']
        if model_name in rnn_model_list:
            return back_model.RNN_Model(seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list, model_name,
                                        self.dataset_name)
        if model_name == 'STAMP':
            return back_model.STAMP(seq_max_len, n_hidden, keep_prob, prediction_embed_list)
        if model_name == 'NARM':
            return back_model.NARM(seq_max_len, n_hidden, keep_prob, prediction_embed_list)

    def get_embedding(self):
        position_copy = tf.tile(self.position_embedding,
                                [tf.shape(self.placeholders['click_label'])[0], 1])  # (bs*seq,embed)
        if self.dataset_name == 'Criteo':
            x1, x2 = tf.split(self.placeholders['input'], [1, 10], 2)
            x2 = tf.cast(x2, tf.int32)
            x2 = tf.nn.embedding_lookup(self.embedding_matrix, x2)
            x2 = tf.reshape(x2, (-1, self.seq_max_len, 10 * self.embedding_size))  # (bs,seq,10*embed)
            if self.time_stamp:
                inputs = tf.concat([x1, x2], axis=-1)  # (bs*seq,10*embed+1)
            else:
                inputs = x2
            if self.position_embed:
                inputs = tf.concat([inputs, tf.reshape(position_copy, (-1, self.seq_max_len, self.embedding_size))],
                                   axis=-1)  # (bs,seq,11*embed+1)
            inputs = tf.transpose(inputs, [1, 0, 2])  # (seq,bs,11*embed+1)
        else:
            inputs = tf.nn.embedding_lookup_sparse(self.embedding_matrix, sp_ids=self.placeholders['input_id'],
                                                   sp_weights=self.placeholders['input_value'])  # (bs*seq,embed)
            if self.position_embed:
                inputs = tf.concat([inputs, position_copy], axis=-1)  # (bs*seq,2*embed)
            inputs = tf.reshape(inputs, (-1, self.seq_max_len, inputs.shape[-1]))  # (bs,seq,2*embed)
            inputs = tf.transpose(inputs, [1, 0, 2])  # (seq,bs,2*embed)
        return inputs

    def forward(self, training=True):
        if training:
            tf.keras.backend.set_learning_phase(1)
        else:
            tf.keras.backend.set_learning_phase(0)
        inputs = self.get_embedding()
        if self.model_name == 'Heroes' or self.model_name == 'Motivate-Heroes':
            prediction_c, prediction_v = self.model(
                tf.concat([inputs, tf.transpose(self.placeholders['click_label'], [1, 0, 2])], axis=-1),
                seq_len=self.placeholders['seq_len'])
        else:
            prediction_c, prediction_v = self.model(inputs, seq_len=self.placeholders['seq_len'])

        if isinstance(prediction_c, list):
            prediction_c, prediction_v = tf.stack(prediction_c), tf.stack(prediction_v)

        '''ops = tf.get_default_graph().get_operations()
        bn_update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, bn_update_ops)'''

        mask = tf.sequence_mask(self.placeholders['seq_len'], self.seq_max_len)
        prediction_c = tf.boolean_mask(tf.transpose(prediction_c, [1, 0, 2]), mask)
        prediction_v = tf.boolean_mask(tf.transpose(prediction_v, [1, 0, 2]), mask)
        reshape_click_label = tf.boolean_mask(self.placeholders['click_label'], mask)
        reshape_conversion_label = tf.boolean_mask(self.placeholders['conversion_label'], mask)

        if training:
            click_loss = self.click_loss(prediction_c, reshape_click_label)
            conversion_loss = self.conversion_loss(prediction_v, reshape_conversion_label)
        else:
            click_loss = tf.losses.log_loss(predictions=prediction_c, labels=reshape_click_label)
            conversion_loss = tf.losses.log_loss(predictions=prediction_v, labels=reshape_conversion_label)
        click_loss = tf.reduce_mean(click_loss)
        conversion_loss = tf.reduce_mean(conversion_loss)
        loss = ((1 - self.ctr_task_wgt) * click_loss + self.ctr_task_wgt * conversion_loss) * 100
        for v in tf.trainable_variables():
            loss += self.l2_reg * tf.nn.l2_loss(v)
        tf.summary.scalar('ctr_loss', click_loss)
        tf.summary.scalar('ctcvr_loss', conversion_loss)

        threshold = 0.5
        one_click = tf.ones_like(reshape_click_label)
        zero_click = tf.zeros_like(reshape_click_label)
        one_cvr = tf.ones_like(reshape_conversion_label)
        zero_cvr = tf.zeros_like(reshape_conversion_label)
        eval_metric_ops = {
            "CTR_AUC": tf.metrics.auc(reshape_click_label, prediction_c),
            "CTR_ACC": tf.metrics.accuracy(reshape_click_label,
                                           tf.where(prediction_c >= threshold, one_click, zero_click)),
            "CTCVR_AUC": tf.metrics.auc(reshape_conversion_label, prediction_v),
            "CTCVR_ACC": tf.metrics.accuracy(reshape_conversion_label,
                                             tf.where(prediction_v >= threshold, one_cvr, zero_cvr))
        }

        #train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        gvs, v = zip(*self.optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)
        train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)
        return click_loss, conversion_loss, loss, eval_metric_ops, train_op
