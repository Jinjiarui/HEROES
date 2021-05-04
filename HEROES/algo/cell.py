import tensorflow as tf
from utils.base import BaseModel
import collections

CellState = collections.namedtuple("CellState", ["c", "h", "conversion_y"])

class Cell(rnn_cell_impl.RNNCell):
    def __init__(self, num_unit, h_below_size, h_above_size, reuse):
        super(Cell, self).__init__(_reuse=reuse)
        self._numunit = num_unit
        self._hbelowsize = h_below_size
        self._habovesize = h_above_size
    
    @property
    def state_size(self):
        # the size of state: c, h, conversion_y
        return (self._numunit, self._numunit, 1)
    
    @property
    def output_size(self):
        # output: h, conversion_y
        return self._numunit+1
    
    def cal_zerostate(self, batch_size):
        c = tf.zeros([batch_size, self._numunit]) # batch_size, hidden_dim
        h = tf.zeros([batch_size, self._numunit]) # batch_size, hidden_dim
        z = tf.zeros([batch_size]) # batch_size, 1
        return CellState(c=c, h=h, z=z)
    
    def cal_cellstate(self, c, g, i, f, z, zb):
        # c, g, i, f: batch_size, hidden_dim
        # conversion_y, zb: batch_size, hidden_dim
        # -> c: batch_size, hidden_dim
        z = tf.squeeze(z, axis=1) # hidden_dim
        zb = tf.squeeze(zb, axis=1) # hidden_dim, 1
        c = tf.where(
            tf.equal(z, tf.constant(1., dtype=tf.float32)),
            tf.multiply(i, g)
            tf.where(
                tf.equal(zb, tf.constant(1., dtype=tf.float32)),
                tf.identity(c),
                tf.add(tf.multiply(f, c), tf.multiply(i, g))
            )
        )
        return c

    def cal_cellhidden(self, h, o, c, z, zb):
        # h, o, c: batch_size, hidden_dim
        # conversion_y, zb: batch_size, 1
        # -> h: batch_size, hidden_dim
        z = tf.squeeze(z, axis=1)
        zb = tf.squeeze(zb, axis=1)
        h = tf.where(
            tf.logical_and(
                tf.equal(z, tf.constant(0., dtype=tf.float32)),
                tf.equal(zb, tf.constant(0., dtype=tf.float32))
            ), # batch_size
            tf.identity(h), # batch_size, hidden_dim, if copy
            tf.multiply(o, tf.tanh(c)) # batch_size, hidden_dim, otherwise
        )
        return h # batch_size, hidden_dim
    
    def cal_indicator(self, z_tilde, slope_multiplier=1):
        sigmoided = tf.sigmoid(z_tilde*slope_multiplier)
        # replace gradient calculation by straight-through estimator
        graph = tf.get_default_graph()
        with tf.name_scope("BinaryRound") as name:
            with graph.gradient_override_map({"Round": "Identity"}):
                z = tf.round(sigmoided, name)
        return tf.squeeze(z, axis=1)
    
    def call(self, inputs, states):
        # inputs: batch_size, hblowsize+1+habovesize
        # states: c: batch_size, hidden_dim, h: batch_size, hidden_dim, conversion_y: batch_size, 1
        # outputs: batch_size, hidden_dim+1
        # new states: c: batch_size, hidden_dim, h: batch_size, hidden_dim, conversion_y: batch_size, 1
        c = states.c
        h = states.h
        z = states.z
        _splits = tf.constant([self._hbelowsize, 1, self._habovesize])
        hb, zb, ha = array_ops.split(
            value=inputs,
            num_or_size_splits=_splits,
            axis=1
        ) # batch_size, hblowsize. batch_size, 1, batch_size, habovesize
        s_recurrent = h # batch_size, hidden_dim
        s_above = tf.multiply(z, ha) # batch_size, habovesize
        s_below = tf.multiply(zb, hb) # batch_size, hblowsize
        states = [s_recurrent, s_above, s_below]
        _lens = 4*self._numunit+1
        # output size: batch_size, 4*hidden_dim+1: i, g, f, o, z_tilde
        _values = tf.layers.dense(states, units=_lens, activation=tf.nn.relu, kernel_initializer=tf.nn.initializers.random_normal())
        _splits = tf.constant(([self._numunit]*4)+[1], dtype=tf.int32)
        i, g, f, o, z_tilde = tf.split(
            value=_values,
            num_or_size_splits=_splits,
            axis=1
        )
        i = tf.sigmoid(i) # batch_size, hidden_dim
        g = tf.tanh(g) # batch_size, hidden_dim
        f = tf.sigmoid(f) # batch_size, hidden_dim
        o = tf.sigmoid(o) # batch_size, hidden_dim
        c = self.cal_cellstate(c, g, i, f, z, zb)
        h = self.cal_cellhidden(h, o, c, z, zb)
        z = tf.expand_dims(self.cal_indicator(z_tilde), -1)
        outputs = tf.concat((h, z), axis=1) # batch_size, hidden_dim+1
        states = CellState(c=c, h=h, z=z)
        return outputs, states
    
class MutliCell(rnn_cell_impl.RNNCell):
    def __init__(self, cells, reuse):
        super(MutliCell, self).__init__(_reuse=reuse)
        self._cells = cells
    
    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self._cells]
    
    @property
    def state_size(self):
        return tuple(cell.state_size for cell in self._cells)
    
    @property
    def output_size(self):
        return self._cells[-1].output_size
    
    def call(self, inputs, states):
        # inputs: batch_size, I+sum(habovesize), I for input: hbelowsize
        # states: list of L length of batch_size, hidden_dim. batch_size, hidden_dim. batch_size, 1
        # hiddens: list of L lenght of batch_size, hidden_dim
        hiddens_size = sum(c._habovesize for c in self._cells)
        _inputs = inputs[:,:-hiddens_size] # batch_size, I
        haboves = inputs[:,-hiddens_size:] # batch_size, sum(habovesize)
        _splits = [c._habovesize for c in self._cells]
        _haboves = tf.split(
            value=haboves,
            num_or_size_splits=_splits,
            axis=1
        ) 
        _zbelow = tf.ones([tf.shape(inputs)[0], 1]) # batch_size, 1
        _inputs = tf.concat([_inputs, _zbelow], axis=1) # batch_size, I+1
        _states = [0]*len(self._cells)
        for _i, _cell in enumerate(self._cells):
            current_state = states[_i] # (batch_size, hidden_dim. batch_size, hidden_dim. batch_size, 1)
            # i=0: batch_size, I+1. batch_size, habovesize -> batch_size, I+1+habovesize
            # i!=0: batch_size, hbelowsize+1. batch_size, habovesize -> batch_size, hbelowsize+1+habovesize.
            current_input = tf.concat([_inputs, _haboves[i]], axis=1)
            _inputs, _state = _cell(current_input, current_state)
            _states[_i] = _state
        hiddens = [s.h for s in _states]
        return hiddens, states
        
