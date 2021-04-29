import tensorflow as tf


class HeroesCell(tf.keras.layers.Layer):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list, **kwargs):
        self.dense_layer = {}
        self.units = units
        self.state_size = [self.units] * state_num
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.prediction_c = [
            tf.keras.layers.Dense(input_dim=units, units=prediction_embed_list[0], use_bias=True,
                                  kernel_initializer='random_normal', name='pc_0')]
        self.prediction_v = [
            tf.keras.layers.Dense(input_dim=units, units=prediction_embed_list[0], use_bias=True,
                                  kernel_initializer='random_normal', name='pv_0')]
        for i in range(1, len(prediction_embed_list)):
            self.prediction_c.append(
                tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                      use_bias=True, kernel_initializer='random_normal', name='pc_{}'.format(i)))
            self.prediction_v.append(
                tf.keras.layers.Dense(input_dim=prediction_embed_list[i - 1], units=prediction_embed_list[i],
                                      use_bias=True, kernel_initializer='random_normal', name='pv_{}'.format(i)))
        self.fc_c = tf.keras.layers.Dense(input_dim=prediction_embed_list[-1], units=n_classes,
                                          use_bias=True, kernel_initializer='random_normal', name='fc_c')
        self.fc_v = tf.keras.layers.Dense(input_dim=prediction_embed_list[-1], units=n_classes,
                                          use_bias=True, kernel_initializer='random_normal', name='fc_v')
        super(HeroesCell, self).__init__(**kwargs)

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

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xgc', 'hgc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xgv', 'hgv']
        dense_layer_name += ['s_c_hat_c', 's_c_hat_v', 's_v_hat_v', 's_v_hat_c']
        return dense_layer_name

    def build(self, input_shape):
        dense_layer_name = self.get_dense_name()
        for i in dense_layer_name:
            if i[0] == 'x':
                self.dense_layer[i] = tf.keras.layers.Dense(units=self.units,
                                                            use_bias=True,
                                                            kernel_initializer='random_normal', name=i)
            else:
                self.dense_layer[i] = tf.keras.layers.Dense(units=self.units,
                                                            use_bias=False,
                                                            kernel_initializer='random_normal', name=i)
        self.built = True

    def call(self, inputs, states):
        inputs, click_label = inputs[:, :-1], inputs[:, -1:]
        h_c = states[0]
        h_v = states[1]
        s_c = states[2]
        s_v = states[3]
        g = states[4]

        s_c_hat = tf.where(g > 0, self.dense_layer['s_c_hat_v'](h_v), self.dense_layer['s_c_hat_c'](h_c))
        f_c = tf.sigmoid(self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](s_c_hat))
        i_c = tf.sigmoid(self.dense_layer['xic'](inputs) + self.dense_layer['hic'](s_c_hat))
        o_c = tf.sigmoid(self.dense_layer['xoc'](inputs) + self.dense_layer['hoc'](s_c_hat))
        g_c = tf.tanh(self.dense_layer['xgc'](inputs) + self.dense_layer['hgc'](s_c_hat))

        s_c = tf.multiply(i_c, g_c) + tf.where(g > 0, tf.zeros_like(s_c), tf.multiply(f_c, s_c))
        h_c = tf.multiply(o_c, tf.tanh(s_c))
        h_c_p = self.predict_call(h_c, 'c')
        g = tf.where(click_label >= 0.5, tf.ones_like(h_c_p), tf.zeros_like(h_c_p))
        g = tf.tile(g, [1, self.units])

        s_v_hat = self.dense_layer['s_v_hat_v'](h_v) + \
                  tf.where(g > 0, self.dense_layer['s_v_hat_c'](h_c), tf.zeros_like(h_v))
        f_v = tf.sigmoid(self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](s_v_hat))
        i_v = tf.sigmoid(self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](s_v_hat))
        o_v = tf.sigmoid(self.dense_layer['xov'](inputs) + self.dense_layer['hov'](s_v_hat))
        g_v = tf.tanh(self.dense_layer['xgv'](inputs) + self.dense_layer['hgv'](s_v_hat))

        s_v = tf.where(g > 0, tf.multiply(f_v, s_v) + tf.multiply(i_v, g_v), s_v)
        h_v = tf.where(g > 0, tf.multiply(o_v, tf.tanh(s_v)), h_v)
        h_v_p = self.predict_call(h_v, 'v') * h_c_p
        return [h_c_p, h_v_p], [h_c, h_v, s_c, s_v, g]


class MotivateHeroesCell(HeroesCell):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list):
        super(MotivateHeroesCell, self).__init__(units, state_num, n_classes, keep_prob, prediction_embed_list)
        self.intensity_fun = lambda x: 5 / (1 + tf.exp(-x / 5))

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc', 'xtc', 'htc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv', 'xtv', 'htv']
        dense_layer_name += ['xfc_hat', 'hfc_hat', 'xic_hat', 'hic_hat', 'xfv_hat', 'hfv_hat', 'xiv_hat', 'hiv_hat']
        dense_layer_name += ['s_c_hat_c', 's_c_hat_v', 's_v_hat_v', 's_v_hat_c']
        return dense_layer_name

    def call(self, inputs, states):
        inputs, click_label = inputs[:, :-1], inputs[:, -1:]
        h_c, h_v, c_c, c_v, c_c_hat, c_v_hat = states[0], states[1], states[2], states[3], states[4], states[5]
        c_t_c, c_t_v = states[6], states[7]
        g = states[8]

        s_c_hat = tf.where(g > 0, self.dense_layer['s_c_hat_v'](h_v), self.dense_layer['s_c_hat_c'](h_c))
        i_c = tf.sigmoid(self.dense_layer['xic'](inputs) + self.dense_layer['hic'](s_c_hat))
        i_c_hat = tf.sigmoid(self.dense_layer['xic_hat'](inputs) + self.dense_layer['hic_hat'](s_c_hat))
        f_c = tf.sigmoid(self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](s_c_hat))
        f_c_hat = tf.sigmoid(self.dense_layer['xfc_hat'](inputs) + self.dense_layer['hfc_hat'](s_c_hat))
        o_c = tf.sigmoid(self.dense_layer['xoc'](inputs) + self.dense_layer['hoc'](s_c_hat))
        z_c = tf.tanh(self.dense_layer['xzc'](inputs) + self.dense_layer['hzc'](s_c_hat))
        t_c = self.intensity_fun(self.dense_layer['xtc'](inputs) + self.dense_layer['htc'](s_c_hat))

        c_c = i_c * z_c + tf.where(g > 0, tf.zeros_like(c_c), f_c * c_t_c)
        c_c_hat = i_c_hat * z_c + tf.where(g > 0, tf.zeros_like(c_c_hat), f_c_hat * c_c_hat)
        c_t_c = c_c_hat + (c_c - c_c_hat) * tf.exp(-t_c)
        h_c = tf.multiply(o_c, tf.tanh(c_t_c))

        h_c_p = self.predict_call(h_c, 'c')
        g = tf.where(click_label >= 0.5, tf.ones_like(h_c_p), tf.zeros_like(h_c_p))
        g = tf.tile(g, [1, self.units])

        s_v_hat = self.dense_layer['s_v_hat_v'](h_v) + \
                  tf.where(g > 0, self.dense_layer['s_v_hat_c'](h_c), tf.zeros_like(h_v))
        i_v = tf.sigmoid(self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](s_v_hat))
        i_v_hat = tf.sigmoid(self.dense_layer['xiv_hat'](inputs) + self.dense_layer['hiv_hat'](s_v_hat))
        f_v = tf.sigmoid(self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](s_v_hat))
        f_v_hat = tf.sigmoid(self.dense_layer['xfv_hat'](inputs) + self.dense_layer['hfv_hat'](s_v_hat))
        o_v = tf.sigmoid(self.dense_layer['xov'](inputs) + self.dense_layer['hov'](s_v_hat))
        z_v = tf.tanh(self.dense_layer['xzv'](inputs) + self.dense_layer['hzv'](s_v_hat))
        t_v = self.intensity_fun(self.dense_layer['xtv'](inputs) + self.dense_layer['htv'](s_v_hat))

        c_v = tf.where(g > 0, f_v * c_t_v + i_v * z_v, c_v)
        c_v_hat = tf.where(g > 0, f_v_hat * c_v_hat + i_v_hat * z_v, c_v_hat)
        c_t_v = c_v_hat + (c_v - c_v_hat) * tf.exp(-t_v)
        h_v = tf.where(g > 0, o_v * tf.tanh(c_t_v), h_v)

        h_v_p = self.predict_call(h_v, 'v') * h_c_p
        return [h_c_p, h_v_p], [h_c, h_v, c_c, c_v, c_c_hat, c_v_hat, c_t_c, c_t_v, g]


class MotivateCell(HeroesCell):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list):
        super(MotivateCell, self).__init__(units, state_num, n_classes, keep_prob, prediction_embed_list)
        self.intensity_fun = lambda x: 5 / (1 + tf.exp(-x / 5))

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc', 'xtc', 'htc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv', 'xtv', 'htv']
        dense_layer_name += ['xfc_hat', 'hfc_hat', 'xic_hat', 'hic_hat', 'xfv_hat', 'hfv_hat', 'xiv_hat', 'hiv_hat']
        return dense_layer_name

    def call(self, inputs, states):
        h_c, h_v, c_c, c_v, c_c_hat, c_v_hat = states[0], states[1], states[2], states[3], states[4], states[5]
        i_c = tf.sigmoid(self.dense_layer['xic'](inputs) + self.dense_layer['hic'](h_c))
        i_c_hat = tf.sigmoid(self.dense_layer['xic_hat'](inputs) + self.dense_layer['hic_hat'](h_c))
        f_c = tf.sigmoid(self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](h_c))
        f_c_hat = tf.sigmoid(self.dense_layer['xfc_hat'](inputs) + self.dense_layer['hfc_hat'](h_c))
        o_c = tf.sigmoid(self.dense_layer['xoc'](inputs) + self.dense_layer['hoc'](h_c))
        z_c = tf.tanh(self.dense_layer['xzc'](inputs) + self.dense_layer['hzc'](h_c))
        t_c = self.intensity_fun(self.dense_layer['xtc'](inputs) + self.dense_layer['htc'](h_c))

        c_t_c = c_c_hat + (c_c - c_c_hat) * tf.exp(-t_c)
        h_c = o_c * tf.tanh(c_t_c)
        c_c = f_c * c_t_c + i_c * z_c
        c_c_hat = f_c_hat * c_c_hat + i_c_hat * z_c

        i_v = tf.sigmoid(self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](h_v))
        i_v_hat = tf.sigmoid(self.dense_layer['xiv_hat'](inputs) + self.dense_layer['hiv_hat'](h_v))
        f_v = tf.sigmoid(self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](h_v))
        f_v_hat = tf.sigmoid(self.dense_layer['xfv_hat'](inputs) + self.dense_layer['hfv_hat'](h_v))
        o_v = tf.sigmoid(self.dense_layer['xov'](inputs) + self.dense_layer['hov'](h_v))
        z_v = tf.tanh(self.dense_layer['xzv'](inputs) + self.dense_layer['hzv'](h_v))
        t_v = self.intensity_fun(self.dense_layer['xtv'](inputs) + self.dense_layer['htv'](h_v))

        c_t_v = c_v_hat + (c_v - c_v_hat) * tf.exp(-t_v)
        h_v = o_v * tf.tanh(c_t_v)
        c_v = f_v * c_t_v + i_v * z_v
        c_v_hat = f_v_hat * c_v_hat + i_v_hat * z_v

        h_c_p = self.predict_call(h_c, target='c')
        h_v_p = self.predict_call(h_v, target='v')
        return [h_c_p, h_c_p * h_v_p], [h_c, h_v, c_c, c_v, c_c_hat, c_v_hat]


class MotivateSingleCell(HeroesCell):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list):
        super(MotivateSingleCell, self).__init__(units, state_num, n_classes, keep_prob, prediction_embed_list)
        self.intensity_fun = lambda x: 5 / (1 + tf.exp(-x / 5))

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc', 'xtc', 'htc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv']
        dense_layer_name += ['xfc_hat', 'hfc_hat', 'xic_hat', 'hic_hat']
        return dense_layer_name

    def call(self, inputs, states):
        h_c, h_v, c_c, c_v, c_c_hat, = states[0], states[1], states[2], states[3], states[4]

        i_c = tf.sigmoid(self.dense_layer['xic'](inputs) + self.dense_layer['hic'](h_c))
        i_c_hat = tf.sigmoid(self.dense_layer['xic_hat'](inputs) + self.dense_layer['hic_hat'](h_c))
        f_c = tf.sigmoid(self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](h_c))
        f_c_hat = tf.sigmoid(self.dense_layer['xfc_hat'](inputs) + self.dense_layer['hfc_hat'](h_c))
        o_c = tf.sigmoid(self.dense_layer['xoc'](inputs) + self.dense_layer['hoc'](h_c))
        z_c = tf.tanh(self.dense_layer['xzc'](inputs) + self.dense_layer['hzc'](h_c))
        t_c = self.intensity_fun(self.dense_layer['xtc'](inputs) + self.dense_layer['htc'](h_c))

        c_t_c = c_c_hat + (c_c - c_c_hat) * tf.exp(-t_c)
        h_c = o_c * tf.tanh(c_t_c)
        c_c = f_c * c_t_c + i_c * z_c
        c_c_hat = f_c_hat * c_c_hat + i_c_hat * z_c

        i_v = tf.sigmoid(self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](h_v))
        f_v = tf.sigmoid(self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](h_v))
        o_v = tf.sigmoid(self.dense_layer['xov'](inputs) + self.dense_layer['hov'](h_v))
        z_v = tf.tanh(self.dense_layer['xzv'](inputs) + self.dense_layer['hzv'](h_v))
        c_v = f_v * c_v + i_v * z_v
        h_v = o_v * tf.tanh(c_v)

        h_c_p = self.predict_call(h_c, target='c')
        h_v_p = self.predict_call(h_v, target='v')
        return [h_c_p, h_c_p * h_v_p], [h_c, h_v, c_c, c_v, c_c_hat]


class SimpleLstmCell(HeroesCell):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list):
        super(SimpleLstmCell, self).__init__(units, state_num, n_classes, keep_prob, prediction_embed_list)
        self.lstm1 = tf.keras.layers.LSTMCell(units)
        self.lstm2 = tf.keras.layers.LSTMCell(units)

    def get_dense_name(self):
        return []

    def call(self, inputs, states):
        h_c, c_c, h_v, c_v = states[0], states[1], states[2], states[3]
        h_c, c_c = self.lstm1(inputs, [h_c, c_c])[1]
        h_v, c_v = self.lstm2(inputs, [h_v, c_v])[1]

        h_c_p = self.predict_call(h_c, target='c')
        h_v_p = self.predict_call(h_v, target='v')
        return [h_c_p, h_c_p * h_v_p], [h_c, h_v, c_c, c_v]


class TimeLstmCell(HeroesCell):
    def __init__(self, units, state_num, n_classes, keep_prob, prediction_embed_list, dataset_name='Criteo'):
        super(TimeLstmCell, self).__init__(units, state_num, n_classes, keep_prob, prediction_embed_list)
        self.dataset_name = dataset_name

    def get_dense_name(self):
        dense_layer_name = ['xfc', 'hfc', 'xic', 'hic', 'xoc', 'hoc', 'xzc', 'hzc',
                            'xfv', 'hfv', 'xiv', 'hiv', 'xov', 'hov', 'xzv', 'hzv']
        dense_layer_name += ['xtc', 'cic', 'ttc', 'toc', 'cfc', 'coc',
                             'xtv', 'civ', 'ttv', 'tov', 'cfv', 'cov']
        return dense_layer_name

    def call(self, inputs, states):
        if self.dataset_name == 'Criteo':
            time_stamp, inputs = inputs[:, 0:1], inputs[:, 1:]
        else:
            inputs, time_stamp = tf.split(inputs, 2, -1)
        print(inputs, time_stamp)
        print('\n\n\n\n\n')
        h_c, c_c, h_v, c_v = states[0], states[1], states[2], states[3]
        i_c = tf.sigmoid(
            self.dense_layer['xic'](inputs) + self.dense_layer['hic'](h_c) + self.dense_layer['cic'](c_c))
        f_c = tf.sigmoid(
            self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](h_c) + self.dense_layer['cfc'](c_c))
        t_c = tf.sigmoid(
            self.dense_layer['xtc'](inputs) + tf.sigmoid(self.dense_layer['ttc'](time_stamp)))
        z_c = tf.tanh(self.dense_layer['xzc'](inputs) + self.dense_layer['hzc'](h_c))

        c_c = f_c * c_c + i_c * t_c * z_c
        o_c = tf.sigmoid(
            self.dense_layer['xoc'](inputs) + self.dense_layer['toc'](time_stamp) +
            self.dense_layer['hoc'](h_c) + self.dense_layer['coc'](c_c))
        h_c = o_c * tf.tanh(c_c)

        i_v = tf.sigmoid(
            self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](h_v) + self.dense_layer['civ'](c_v))
        f_v = tf.sigmoid(
            self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](h_v) + self.dense_layer['cfv'](c_v))
        t_v = tf.sigmoid(
            self.dense_layer['xtv'](inputs) + tf.sigmoid(self.dense_layer['ttv'](time_stamp)))
        z_v = tf.tanh(self.dense_layer['xzv'](inputs) + self.dense_layer['hzv'](h_v))

        c_v = f_v * c_v + i_v * t_v * z_v
        o_v = tf.sigmoid(
            self.dense_layer['xov'](inputs) + self.dense_layer['tov'](time_stamp) +
            self.dense_layer['hov'](h_v) + self.dense_layer['cov'](c_v))
        h_v = o_v * tf.tanh(c_v)

        h_c_p = self.predict_call(h_c, target='c')
        h_v_p = self.predict_call(h_v, target='v')
        return [h_c_p, h_c_p * h_v_p], [h_c, h_v, c_c, c_v]


class RNN_Model(tf.keras.layers.Layer):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list, model_name='Heroes',
                 dataset_name='Criteo'):
        super(RNN_Model, self).__init__()
        self.seq_max_len = tf.range(seq_max_len)
        if model_name == 'RRN':
            self.cell = SimpleLstmCell(n_hidden, 4, n_classes, keep_prob, prediction_embed_list)
        elif model_name == 'Heroes':
            self.cell = HeroesCell(n_hidden, 5, n_classes, keep_prob, prediction_embed_list)
        elif model_name == 'Motivate-Heroes':
            self.cell = MotivateHeroesCell(n_hidden, 7, n_classes, keep_prob, prediction_embed_list)
        elif model_name == 'motivate':
            self.cell = MotivateCell(n_hidden, 6, n_classes, keep_prob, prediction_embed_list)
        elif model_name == 'motivate-single':
            self.cell = MotivateSingleCell(n_hidden, 5, n_classes, keep_prob, prediction_embed_list)
        elif model_name == 'time_LSTM':
            self.cell = TimeLstmCell(n_hidden, 4, n_classes, keep_prob, prediction_embed_list, dataset_name)

        self.layer = tf.keras.layers.RNN(self.cell, name="RNN", trainable=True, time_major=True, return_sequences=True)

    def call(self, inputs, **kwargs):
        seq_len = kwargs['seq_len']
        mask = tf.expand_dims(tf.transpose(tf.map_fn(lambda i: self.seq_max_len < i, seq_len, tf.bool)), dim=-1)
        y = self.layer(inputs, mask=mask)
        return y


class STAMP(tf.keras.layers.Layer):
    def __init__(self, seq_max_len, n_hidden, keep_prob, prediction_embed_list):
        super(STAMP, self).__init__()
        self.n_hidden = n_hidden
        self.seq_max_len = seq_max_len
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.w0 = tf.keras.layers.Dense(input_dim=n_hidden, units=1, use_bias=False,
                                        kernel_initializer='random_normal', name='w1')
        self.w1 = tf.keras.layers.Dense(units=n_hidden, use_bias=True,
                                        kernel_initializer='random_normal', name='w1')
        self.w2 = tf.keras.layers.Dense(units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='w2')
        self.w3 = tf.keras.layers.Dense(units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='w3')
        self.prediction_c = [
            [tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0'),
             tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0_')
             ]]
        self.prediction_v = [
            [tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pv_0'),
             tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
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
    def __init__(self, seq_max_len, n_hidden, keep_prob, prediction_embed_list):
        super(NARM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_max_len = seq_max_len
        self.drop_out = tf.keras.layers.Dropout(rate=1 - keep_prob)
        self.activate = tf.keras.layers.LeakyReLU()
        self.A1 = tf.keras.layers.Dense(input_dim=n_hidden, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='A1')
        self.A2 = tf.keras.layers.Dense(input_dim=n_hidden, units=n_hidden, use_bias=False,
                                        kernel_initializer='random_normal', name='A2')
        self.v = tf.keras.layers.Dense(input_dim=n_hidden, units=1, use_bias=False,
                                       kernel_initializer='random_normal', name='v')

        self.prediction_c = [
            [tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0'),
             tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pc_0_')
             ]]
        self.prediction_v = [
            [tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
                                   kernel_initializer='random_normal', name='pv_0'),
             tf.keras.layers.Dense(units=prediction_embed_list[0], use_bias=True,
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

        self.GRU_g = tf.keras.layers.GRU(units=n_hidden, return_sequences=True, time_major=True)
        self.GRU_l = tf.keras.layers.GRU(units=n_hidden, return_sequences=True, time_major=True)

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
        h_g = self.GRU_g(inputs)
        h_l = self.GRU_l(inputs)
        h_l_1 = self.A1(h_l)
        h_l_2 = self.A2(h_l)
        c_l = tf.map_fn(lambda i: tf.reduce_sum(
            self.v(tf.sigmoid(h_l_1[:i + 1] + h_l_2[i:i + 1])) * h_l[:i + 1], axis=0),
                        tf.range(self.seq_max_len), dtype=inputs.dtype)
        c_t = tf.concat([h_g, c_l], axis=-1)
        prediction_c = self.predict_call([c_t, inputs], 'c')
        prediction_v = self.predict_call([c_t, inputs], 'v')
        prediction_v *= prediction_c
        return prediction_c, prediction_v
