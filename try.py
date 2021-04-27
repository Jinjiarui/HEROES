import tensorflow as tf


class HeroesCell(tf.keras.layers.Layer):
    def __init__(self, units, n_classes, keep_prob, prediction_embed_list, **kwargs):
        self.units = units
        self.state_size = [self.units, self.units, self.units, self.units, self.units]
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
        self.dense_layer = {}
        for i in dense_layer_name:
            if i[0] == 'x':
                self.dense_layer[i] = tf.keras.layers.Dense(input_dim=input_shape[-1] - 1, units=self.units,
                                                            use_bias=True,
                                                            kernel_initializer='random_normal', name=i)
            else:
                self.dense_layer[i] = tf.keras.layers.Dense(input_dim=self.units, units=self.units,
                                                            use_bias=False,
                                                            kernel_initializer='random_normal', name=i)
        self.built = True

    def call(self, inputs, states):
        inputs, click_label = inputs[:, :-1], inputs[:, -1:]
        H_c = states[0]
        H_v = states[1]
        s_c = states[2]
        s_v = states[3]
        g = states[4]

        f_c = tf.sigmoid(self.dense_layer['xfc'](inputs) + self.dense_layer['hfc'](H_c))
        i_c = tf.sigmoid(self.dense_layer['xic'](inputs) + self.dense_layer['hic'](H_c))
        o_c = tf.sigmoid(self.dense_layer['xoc'](inputs) + self.dense_layer['hoc'](H_c))
        g_c = tf.tanh(self.dense_layer['xgc'](inputs) + self.dense_layer['hgc'](H_c))
        s_c_hat = tf.tanh(tf.multiply(1 - g, self.dense_layer['s_c_hat_c'](H_c)) \
                          + tf.multiply(g, self.dense_layer['s_c_hat_v'](H_v)))
        s_c = s_c_hat + tf.multiply(i_c, g_c) + tf.multiply(1 - g, tf.multiply(f_c, s_c))
        H_c = tf.multiply(o_c, tf.tanh(s_c))
        H_c_p = self.predict_call(H_c, 'c')
        g = tf.where(click_label >= 0.5, tf.ones_like(H_c_p), tf.zeros_like(H_c_p))
        g = tf.tile(g, [1, self.units])
        f_v = tf.sigmoid(self.dense_layer['xfv'](inputs) + self.dense_layer['hfv'](H_v))
        i_v = tf.sigmoid(self.dense_layer['xiv'](inputs) + self.dense_layer['hiv'](H_v))
        o_v = tf.sigmoid(self.dense_layer['xov'](inputs) + self.dense_layer['hov'](H_v))
        g_v = tf.tanh(self.dense_layer['xgv'](inputs) + self.dense_layer['hgv'](H_v))
        s_v_hat = tf.tanh(self.dense_layer['s_v_hat_v'](H_v) \
                          + tf.multiply(g, self.dense_layer['s_v_hat_c'](H_c)))
        s_v = s_v_hat + tf.multiply(1 - g, s_v) + tf.multiply(g, tf.multiply(f_v, s_v) + tf.multiply(i_v, g_v))
        H_v = tf.multiply(1 - g, H_v) + tf.multiply(g, tf.multiply(o_v, tf.tanh(s_v)))
        H_v_p = self.predict_call(H_v, 'v') * H_c_p
        return [H_c_p, H_v_p], [H_c, H_v, s_c, s_v, g]


class RNN_model(tf.keras.layers.Layer):
    def __init__(self, seq_max_len, n_hidden, n_classes, keep_prob, prediction_embed_list):
        super(RNN_model, self).__init__()
        self.seq_max_len = tf.range(seq_max_len)
        self.cell = HeroesCell(n_hidden, n_classes, keep_prob, prediction_embed_list)
        self.layer = tf.keras.layers.RNN(self.cell, name="RNN", trainable=True, time_major=True, return_sequences=True)

    def call(self, inputs, **kwargs):
        seq_len = kwargs['seq_len']
        mask = tf.expand_dims(tf.transpose(tf.map_fn(lambda i: self.seq_max_len < i, seq_len, tf.bool)), dim=-1)
        y = self.layer(inputs1, mask=mask)
        return y


seq_max_len = 11
inputs1 = tf.random_normal((seq_max_len, 10, 10))

seq_len = tf.convert_to_tensor([5, 4, 6, 1, 4, 7, 8, 8, 1, seq_max_len])

model = RNN_model(seq_max_len, n_hidden=30, n_classes=1, keep_prob=0.6, prediction_embed_list=[64, 33, 17])

pc, pv = model(inputs1, seq_len=seq_len)

print(pc, pv)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10):
        print(sess.run(pc[:, i, :]).squeeze())
print(pc, pv)
