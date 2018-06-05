from tensorflow.python.ops.rnn_cell import *
class CustomLSTMCell(RNNCell):
    def __init__(self, num_units, forget_bias = 1.0, activation = tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
    @property
    def state_size(self):
        return 2 * self._num_units
    @property
    def output_size(self):
        return self._num_units
    
    def __call__(self, inputs, state, reward, scope = None):
        with vs.variable_scope(scope or type(self).__name__):
            c,h,action = array_ops.split(1,3, state)
            concat = _linear([inputs, action, reward, h], 4*self._num_units, True)
            i,j,f,o = array_ops.split(1, 4, concat)
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)
            new_state = array_ops.concat(1, [new_c, new_h, o])
            return new_h, new_state
    
