from __future__ import absolute_import
from keras.layers import Dense
import keras.backend as K


class Dense_tied(Dense):
    """
    A fully connected layer with tied weights.

    For further information, refer to https://github.com/hugochan/KATE
    """

    def __init__(self, units,
                 activation=None, use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 tied_to=None, **kwargs):
        self.tied_to = tied_to

        super(Dense_tied, self).__init__(units=units,
                                         activation=activation, use_bias=use_bias,
                                         bias_initializer=bias_initializer,
                                         kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                         **kwargs)

    def build(self, input_shape):
        super(Dense_tied, self).build(input_shape)  # be sure you call this somewhere!
        if self.kernel in self.trainable_weights:
            self.trainable_weights.remove(self.kernel)

    def call(self, x, mask=None):
        # Use tied weights
        self.kernel = K.transpose(self.tied_to.kernel)
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output += self.bias
        return self.activation(output)
