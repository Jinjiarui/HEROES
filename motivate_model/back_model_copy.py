import tensorflow as tf


class Heroes(tf.keras.layers.Layer):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list):
        super(Heroes, self).__init__()
        self.n_hidden = n_hidden
        self.seq_max_len = seq_max_len
        dense_layer_name = self.get_dense_name()
        self.dense_layer = {}
        for i in dense_layer_name:
            if i[0] == 'x':
                self.dense_layer[i] = tf.keras.layers.Dense(units=n_hidden,
                                                            use_bias=True,
                                                            kernel_initializer='random_normal', name=i)
            else:
                self.dense_layer[i] = tf.keras.layers.Dense(input_dim=n_hidden, units=n_hidden,
                                                            use_bias=False,
                                                            kernel_initializer='random_normal', name=i)
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.prediction_c = [
            tf.keras.layers.Dense(input_dim=n_hidden, units=prediction_embed_list[0], use_bias=True,
                                  kernel_initializer='random_normal', name='pc_0')]
        self.prediction_v = [
            tf.keras.layers.Dense(input_dim=n_hidden, units=prediction_embed_list[0], use_bias=True,
                                  kernel_initializer='random_normal', name='pv_0')]
        for i in range(1, len(prediction_embed_list)):
            self.prediction_c.append(
                tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                      use_bias=True, kernel_initializer='random_normal', name='pc_{}'.format(i)))
            self.prediction_v.append(
                tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                      use_bias=True, kernel_initializer='random_normal', name='pv_{}'.format(i)))
        self.fc_c = tf.keras.layers.Dense(input_dim=prediction_embed_list[- 1], units=n_classes,
                                          use_bias=True, kernel_initializer='random_normal', name='fc_c')
        self.fc_v = tf.keras.layers.Dense(input_dim=prediction_embed_list[- 1], units=n_classes,
                                          use_bias=True, kernel_initializer='random_normal', name='fc_v')

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xgc', 'hgc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xgv', 'hgv']
        dense_layer_name += ['s_c_hat_c', 's_c_hat_v', 's_v_hat_v', 's_v_hat_c']
        return dense_layer_name

    def predict_call(self, inputs, target):
        if target == 'c':
            predict_layers = self.prediction_c
            fc = self.fc_c
        else:
            predict_layers = self.prediction_v
            fc = self.fc_v
        for i in range(len(predict_layers)):
            inputs = predict_layers[i](inputs)
            inputs = self.activate(inputs)
            inputs = self.drop_out(inputs)
        return tf.sigmoid(fc(inputs))

    def call(self, inputs, **kwargs):
        click_label = None
        if isinstance(inputs, list):
            inputs, click_label = inputs
            click_label = tf.transpose(click_label, [1, 0, 2])  # (seq,bs,n_class)
        with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            H_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            H_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            s_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            s_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            prediction_c = []
            prediction_v = []
            g = self.predict_call(H_c, 'c')
            pc = tf.ones_like(g)  # The product of 1-H_c
            pv = tf.ones_like(g)  # The product of 1-H_v
            g = tf.tile(g, [1, self.n_hidden])  # (bs,hidden)
            for i in range(self.seq_max_len):
                f_c = tf.sigmoid(self.dense_layer['xfc'](inputs[i]) + self.dense_layer['hfc'](H_c))
                i_c = tf.sigmoid(self.dense_layer['xic'](inputs[i]) + self.dense_layer['hic'](H_c))
                o_c = tf.sigmoid(self.dense_layer['xoc'](inputs[i]) + self.dense_layer['hoc'](H_c))
                g_c = tf.tanh(self.dense_layer['xgc'](inputs[i]) + self.dense_layer['hgc'](H_c))
                s_c_hat = tf.tanh(tf.multiply(1 - g, self.dense_layer['s_c_hat_c'](H_c)) \
                                  + tf.multiply(g, self.dense_layer['s_c_hat_v'](H_v)))
                s_c = s_c_hat + tf.multiply(i_c, g_c) + tf.multiply(1 - g, tf.multiply(f_c, s_c))
                H_c = tf.multiply(o_c, tf.tanh(s_c))
                H_c_p = self.predict_call(H_c, 'c')
                prediction_c.append(H_c_p)
                if click_label is not None:
                    g = tf.where(click_label[i] >= 0.5, tf.ones_like(prediction_c[-1]), tf.zeros_like(prediction_c[-1]))
                    pc = tf.where(click_label[i] >= 0.5, tf.ones_like(prediction_c[-1]), tf.multiply(1 - H_c_p, pc))
                else:
                    g = H_c_p
                    pc = tf.where(prediction_c[-1] >= 0.5, tf.ones_like(prediction_c[-1]), tf.multiply(1 - H_c_p, pc))
                g = tf.tile(g, [1, self.n_hidden])

                f_v = tf.sigmoid(self.dense_layer['xfv'](inputs[i]) + self.dense_layer['hfv'](H_v))
                i_v = tf.sigmoid(self.dense_layer['xiv'](inputs[i]) + self.dense_layer['hiv'](H_v))
                o_v = tf.sigmoid(self.dense_layer['xov'](inputs[i]) + self.dense_layer['hov'](H_v))
                g_v = tf.tanh(self.dense_layer['xgv'](inputs[i]) + self.dense_layer['hgv'](H_v))
                s_v_hat = tf.tanh(self.dense_layer['s_v_hat_v'](H_v) \
                                  + tf.multiply(g, self.dense_layer['s_v_hat_c'](H_c)))
                s_v = s_v_hat + tf.multiply(1 - g, s_v) + tf.multiply(g, tf.multiply(f_v, s_v) + tf.multiply(i_v, g_v))
                H_v = tf.multiply(1 - g, H_v) + tf.multiply(g, tf.multiply(o_v, tf.tanh(s_v)))
                H_v_p = self.predict_call(H_v, 'v') * H_c_p
                prediction_v.append(H_v_p)
                if click_label is not None:
                    pv = tf.where(click_label[-1] >= 0.5, tf.ones_like(prediction_v[-1]), tf.multiply(1 - H_v_p, pv))
                else:
                    pv = tf.where(prediction_c[-1] >= 0.5, tf.ones_like(prediction_v[-1]), tf.multiply(1 - H_v_p, pv))
        return prediction_c, prediction_v


class motivate_model(Heroes):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list):
        super(motivate_model, self).__init__(seq_max_len, n_hidden, n_classes, keep_prob,
                                             prediction_embed_list)
        self.intensity_fun = lambda x: 5 / (1 + tf.exp(-x / 5))

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc', 'xtc', 'htc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv', 'xtv', 'htv']
        dense_layer_name += ['xfc_hat', 'hfc_hat', 'xic_hat', 'hic_hat', 'xfv_hat', 'hfv_hat', 'xiv_hat', 'hiv_hat']
        return dense_layer_name

    def call(self, inputs, **kwargs):
        with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            h_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            h_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            c_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_c_hat = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_v_hat = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            prediction_c = []
            prediction_v = []
            for i in range(self.seq_max_len):
                i_c = tf.sigmoid(self.dense_layer['xic'](inputs[i]) + self.dense_layer['hic'](h_c))
                i_c_hat = tf.sigmoid(self.dense_layer['xic_hat'](inputs[i]) + self.dense_layer['hic_hat'](h_c))
                f_c = tf.sigmoid(self.dense_layer['xfc'](inputs[i]) + self.dense_layer['hfc'](h_c))
                f_c_hat = tf.sigmoid(self.dense_layer['xfc_hat'](inputs[i]) + self.dense_layer['hfc_hat'](h_c))
                o_c = tf.sigmoid(self.dense_layer['xoc'](inputs[i]) + self.dense_layer['hoc'](h_c))
                z_c = tf.tanh(self.dense_layer['xzc'](inputs[i]) + self.dense_layer['hzc'](h_c))
                t_c = self.intensity_fun(self.dense_layer['xtc'](inputs[i]) + self.dense_layer['htc'](h_c))

                c_t_c = c_c_hat + (c_c - c_c_hat) * tf.exp(-t_c)
                h_c = o_c * tf.tanh(c_t_c)
                c_c = f_c * c_t_c + i_c * z_c
                c_c_hat = f_c_hat * c_c_hat + i_c_hat * z_c

                i_v = tf.sigmoid(self.dense_layer['xiv'](inputs[i]) + self.dense_layer['hiv'](h_v))
                i_v_hat = tf.sigmoid(self.dense_layer['xiv_hat'](inputs[i]) + self.dense_layer['hiv_hat'](h_v))
                f_v = tf.sigmoid(self.dense_layer['xfv'](inputs[i]) + self.dense_layer['hfv'](h_v))
                f_v_hat = tf.sigmoid(self.dense_layer['xfv_hat'](inputs[i]) + self.dense_layer['hfv_hat'](h_v))
                o_v = tf.sigmoid(self.dense_layer['xov'](inputs[i]) + self.dense_layer['hov'](h_v))
                z_v = tf.tanh(self.dense_layer['xzv'](inputs[i]) + self.dense_layer['hzv'](h_v))
                t_v = self.intensity_fun(self.dense_layer['xtv'](inputs[i]) + self.dense_layer['htv'](h_v))

                c_t_v = c_v_hat + (c_v - c_v_hat) * tf.exp(-t_v)
                h_v = o_v * tf.tanh(c_t_v)
                c_v = f_v * c_t_v + i_v * z_v
                c_v_hat = f_v_hat * c_v_hat + i_v_hat * z_v

                h_c_p = self.predict_call(h_c, target='c')
                h_v_p = self.predict_call(h_v, target='v')
                prediction_c.append(h_c_p)
                prediction_v.append(h_c_p * h_v_p)
        return prediction_c, prediction_v


class motivate_single(motivate_model):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list):
        super(motivate_single, self).__init__(seq_max_len, n_hidden, n_classes, keep_prob,
                                              prediction_embed_list)

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc', 'xtc', 'htc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv']
        dense_layer_name += ['xfc_hat', 'hfc_hat', 'xic_hat', 'hic_hat']
        return dense_layer_name

    def call(self, inputs, **kwargs):
        with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            h_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            h_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            c_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_c_hat = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            prediction_c = []
            prediction_v = []
            for i in range(self.seq_max_len):
                i_c = tf.sigmoid(self.dense_layer['xic'](inputs[i]) + self.dense_layer['hic'](h_c))
                i_c_hat = tf.sigmoid(self.dense_layer['xic_hat'](inputs[i]) + self.dense_layer['hic_hat'](h_c))
                f_c = tf.sigmoid(self.dense_layer['xfc'](inputs[i]) + self.dense_layer['hfc'](h_c))
                f_c_hat = tf.sigmoid(self.dense_layer['xfc_hat'](inputs[i]) + self.dense_layer['hfc_hat'](h_c))
                o_c = tf.sigmoid(self.dense_layer['xoc'](inputs[i]) + self.dense_layer['hoc'](h_c))
                z_c = tf.tanh(self.dense_layer['xzc'](inputs[i]) + self.dense_layer['hzc'](h_c))
                t_c = self.intensity_fun(self.dense_layer['xtc'](inputs[i]) + self.dense_layer['htc'](h_c))

                c_t_c = c_c_hat + (c_c - c_c_hat) * tf.exp(-t_c)
                h_c = o_c * tf.tanh(c_t_c)
                c_c = f_c * c_t_c + i_c * z_c
                c_c_hat = f_c_hat * c_c_hat + i_c_hat * z_c

                i_v = tf.sigmoid(self.dense_layer['xiv'](inputs[i]) + self.dense_layer['hiv'](h_v))
                f_v = tf.sigmoid(self.dense_layer['xfv'](inputs[i]) + self.dense_layer['hfv'](h_v))
                o_v = tf.sigmoid(self.dense_layer['xov'](inputs[i]) + self.dense_layer['hov'](h_v))
                z_v = tf.tanh(self.dense_layer['xzv'](inputs[i]) + self.dense_layer['hzv'](h_v))
                c_v = f_v * c_v + i_v * z_v
                h_v = o_v * tf.tanh(c_v)

                h_c_p = self.predict_call(h_c, target='c')
                h_v_p = self.predict_call(h_v, target='v')
                prediction_c.append(h_c_p)
                prediction_v.append(h_c_p * h_v_p)
        return prediction_c, prediction_v


class LSTM(Heroes):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list):
        super(LSTM, self).__init__(seq_max_len, n_hidden, n_classes, keep_prob,
                                   prediction_embed_list)

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv']
        return dense_layer_name

    def call(self, inputs, **kwargs):
        with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            h_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            h_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            c_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            prediction_c = []
            prediction_v = []
            for i in range(self.seq_max_len):
                i_c = tf.sigmoid(self.dense_layer['xic'](inputs[i]) + self.dense_layer['hic'](h_c))
                f_c = tf.sigmoid(self.dense_layer['xfc'](inputs[i]) + self.dense_layer['hfc'](h_c))
                o_c = tf.sigmoid(self.dense_layer['xoc'](inputs[i]) + self.dense_layer['hoc'](h_c))
                z_c = tf.tanh(self.dense_layer['xzc'](inputs[i]) + self.dense_layer['hzc'](h_c))

                c_c = f_c * c_c + i_c * z_c
                h_c = o_c * tf.tanh(c_c)

                i_v = tf.sigmoid(self.dense_layer['xiv'](inputs[i]) + self.dense_layer['hiv'](h_v))
                f_v = tf.sigmoid(self.dense_layer['xfv'](inputs[i]) + self.dense_layer['hfv'](h_v))
                o_v = tf.sigmoid(self.dense_layer['xov'](inputs[i]) + self.dense_layer['hov'](h_v))
                z_v = tf.tanh(self.dense_layer['xzv'](inputs[i]) + self.dense_layer['hzv'](h_v))
                c_v = f_v * c_v + i_v * z_v
                h_v = o_v * tf.tanh(c_v)

                h_c_p = self.predict_call(h_c, target='c')
                h_v_p = self.predict_call(h_v, target='v')
                prediction_c.append(h_c_p)
                prediction_v.append(h_c_p * h_v_p)
        return prediction_c, prediction_v


class time_LSTM(Heroes):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list,
                 dataset_name):
        super(time_LSTM, self).__init__(seq_max_len, n_hidden, n_classes, keep_prob,
                                        prediction_embed_list)
        self.dataset_name = dataset_name

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv']
        dense_layer_name += ['xtc', 'cic', 'ttc', 'toc', 'cfc', 'coc',
                             'xtv', 'civ', 'ttv', 'tov', 'cfv', 'cov']
        return dense_layer_name

    def call(self, inputs, **kwargs):
        if self.dataset_name == 'Criteo':
            time_stamp, inputs = inputs[:, :, 0], inputs[:, :, 1:]
            time_stamp = tf.expand_dims(time_stamp, -1)
        else:
            inputs, time_stamp = tf.split(inputs, 2, 2)
        with tf.name_scope('RNN'), tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):
            h_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            h_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))  # (bs,hidden)
            c_c = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            c_v = tf.zeros(shape=(tf.shape(inputs)[1], self.n_hidden))
            prediction_c = []
            prediction_v = []
            for i in range(self.seq_max_len):
                i_c = tf.sigmoid(
                    self.dense_layer['xic'](inputs[i]) + self.dense_layer['hic'](h_c) + self.dense_layer['cic'](c_c))
                f_c = tf.sigmoid(
                    self.dense_layer['xfc'](inputs[i]) + self.dense_layer['hfc'](h_c) + self.dense_layer['cfc'](c_c))
                t_c = tf.sigmoid(
                    self.dense_layer['xtc'](inputs[i]) + tf.sigmoid(self.dense_layer['ttc'](time_stamp[i])))
                z_c = tf.tanh(self.dense_layer['xzc'](inputs[i]) + self.dense_layer['hzc'](h_c))

                c_c = f_c * c_c + i_c * t_c * z_c
                o_c = tf.sigmoid(
                    self.dense_layer['xoc'](inputs[i]) + self.dense_layer['toc'](time_stamp[i]) +
                    self.dense_layer['hoc'](h_c) + self.dense_layer['coc'](c_c))
                h_c = o_c * tf.tanh(c_c)

                i_v = tf.sigmoid(
                    self.dense_layer['xiv'](inputs[i]) + self.dense_layer['hiv'](h_v) + self.dense_layer['civ'](c_v))
                f_v = tf.sigmoid(
                    self.dense_layer['xfv'](inputs[i]) + self.dense_layer['hfv'](h_v) + self.dense_layer['cfv'](c_v))
                t_v = tf.sigmoid(
                    self.dense_layer['xtv'](inputs[i]) + tf.sigmoid(self.dense_layer['ttv'](time_stamp[i])))
                z_v = tf.tanh(self.dense_layer['xzv'](inputs[i]) + self.dense_layer['hzv'](h_v))

                c_v = f_v * c_v + i_v * t_v * z_v
                o_v = tf.sigmoid(
                    self.dense_layer['xov'](inputs[i]) + self.dense_layer['tov'](time_stamp[i]) +
                    self.dense_layer['hov'](h_v) + self.dense_layer['cov'](c_v))
                h_v = o_v * tf.tanh(c_v)

                h_c_p = self.predict_call(h_c, target='c')
                h_v_p = self.predict_call(h_v, target='v')
                prediction_c.append(h_c_p)
                prediction_v.append(h_c_p * h_v_p)
        return prediction_c, prediction_v


class STAMP(tf.keras.layers.Layer):
    def __init__(self, embedding_size, seq_max_len, n_hidden, keep_prob, prediction_embed_list):
        super(STAMP, self).__init__()
        self.n_hidden = n_hidden
        self.seq_max_len = seq_max_len
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.w0 = tf.keras.layers.Dense(input_dim=n_hidden, units=1, use_bias=False,
                                        kernel_initializer='random_normal', name='w1')
        self.w1 = tf.keras.layers.Dense(input_dim=embedding_size, units=n_hidden, use_bias=True,
                                        kernel_initializer='random_normal', name='w1')
        self.w2 = tf.keras.layers.Dense(input_dim=embedding_size, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='w2')
        self.w3 = tf.keras.layers.Dense(input_dim=embedding_size, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='w3')
        self.prediction_c = [
            [tf.keras.layers.Dense(input_dim=embedding_size, units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0'),
             tf.keras.layers.Dense(input_dim=embedding_size, units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0_')
             ]]
        self.prediction_v = [
            [tf.keras.layers.Dense(input_dim=n_hidden, units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pv_0'),
             tf.keras.layers.Dense(input_dim=n_hidden, units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pv_0_')]
        ]
        for i in range(1, len(prediction_embed_list)):
            self.prediction_c.append(
                [tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                       use_bias=True, kernel_initializer='random_normal', name='pc_{}'.format(i)),
                 tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                       use_bias=True, kernel_initializer='random_normal', name='pc_{}_'.format(i)),
                 ])
            self.prediction_v.append(
                [tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                       use_bias=True, kernel_initializer='random_normal', name='pv_{}'.format(i)),
                 tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                       use_bias=True, kernel_initializer='random_normal', name='pv_{}_'.format(i)),
                 ])

    def predict_call(self, inputs, target):
        inputs1, inputs2 = inputs
        if target == 'c':
            predict_layers = self.prediction_c
        else:
            predict_layers = self.prediction_v
        for i in range(len(predict_layers)):
            inputs1 = predict_layers[i][0](inputs1)
            inputs1 = self.activate(inputs1)
            inputs1 = self.drop_out(inputs1)

            inputs2 = predict_layers[i][1](inputs2)
            inputs2 = self.activate(inputs2)
            inputs2 = self.drop_out(inputs2)
        return tf.sigmoid(tf.reduce_sum(inputs1 * inputs2, -1, keep_dims=True))

    def call(self, inputs, **kwargs):
        m_s = tf.cumsum(inputs) / tf.reshape(tf.range(self.seq_max_len, dtype=tf.float32) + 1, (-1, 1, 1))
        x1 = self.w1(inputs)
        x2 = self.w2(inputs)
        wms = self.w3(m_s)
        m_a = tf.map_fn(lambda i: tf.reduce_sum(
            self.w0(tf.sigmoid(x1[:i + 1] + x2[i:i + 1] + wms[i:i + 1])) * inputs[:i + 1], axis=0),
                        tf.range(self.seq_max_len), dtype=inputs.dtype)
        prediction_c = self.predict_call([m_a, inputs], 'c')
        prediction_v = self.predict_call([m_a, inputs], 'v')
        prediction_v *= prediction_c
        return prediction_c, prediction_v


class NARM(tf.keras.layers.Layer):

    def __init__(self, embedding_size, seq_max_len, n_hidden, keep_prob):
        super(NARM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_max_len = seq_max_len
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.A1 = tf.keras.layers.Dense(input_dim=n_hidden, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='A1')
        self.A2 = tf.keras.layers.Dense(input_dim=n_hidden, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='A2')
        self.GRU_c = tf.keras.layers.GRU(units=n_hidden, input_dim=embedding_size, return_sequences=True,
                                         time_major=True)
        self.GRU_v = tf.keras.layers.GRU(units=n_hidden, input_dim=embedding_size, return_sequences=True,
                                         time_major=True)

    def call(self, inputs, **kwargs):
        h_c = self.GRU_c(inputs)
