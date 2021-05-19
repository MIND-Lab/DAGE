from __future__ import absolute_import
from keras.engine import Layer
import keras.backend as K
import tensorflow as tf
import warnings


class KCompetitive(Layer):
    '''Applies K-Competitive layer.

    For further information, refer to https://github.com/hugochan/KATE

    # Arguments
    '''

    def __init__(self, topk, ctype, factor, **kwargs):
        self.topk = topk
        self.ctype = ctype
        self.uses_learning_phase = True
        self.supports_masking = True
        self.factor = factor
        super(KCompetitive, self).__init__(**kwargs)

    def call(self, x):
        if self.ctype == 'ksparse':
            return K.in_train_phase(self.kSparse(x, self.topk), x)
        elif self.ctype == 'kcomp':
            return K.in_train_phase(self.k_comp_tanh(x, self.topk), x)
        else:
            return x

    def get_config(self):
        config = {'topk': self.topk, 'ctype': self.ctype}
        base_config = super(KCompetitive, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def k_comp_tanh(self, x, topk):
        factor = self.factor
        print('run k_comp_tanh')
        dim = int(x.shape[1])
        print(x.shape)
        print(x)
        # batch_size = tf.to_float(tf.shape(x)[0])
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        P = (x + tf.abs(x)) / 2
        N = (x - tf.abs(x)) / 2

        values, indices = tf.nn.top_k(P, int(
            topk / 2))  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]
        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, int(topk / 2)])  # will be [[0, 0], [1, 1]]
        full_indices = tf.stack([my_range_repeated, indices],
                                axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])
        P_reset = tf.compat.v1.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
                                               validate_indices=False)

        values2, indices2 = tf.nn.top_k(-N, topk - int(topk / 2))
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices2)[0]), 1)
        my_range_repeated = tf.tile(my_range, [1, topk - int(topk / 2)])
        full_indices2 = tf.stack([my_range_repeated, indices2], axis=2)
        full_indices2 = tf.reshape(full_indices2, [-1, 2])
        N_reset = tf.compat.v1.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(values2, [-1]), default_value=0.,
                                               validate_indices=False)

        # 1)
        # res = P_reset - N_reset
        # tmp = 1 * batch_size * tf.reduce_sum(x - res, 1, keep_dims=True) / topk

        # P_reset = tf.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)
        # N_reset = tf.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, tf.abs(tmp)), [-1]), default_value=0., validate_indices=False)

        # 2)
        # factor = 0.
        # factor = 2. / topk
        P_tmp = factor * tf.reduce_sum(P - P_reset, 1, keepdims=True)  # 6.26
        N_tmp = factor * tf.reduce_sum(-N - N_reset, 1, keepdims=True)
        P_reset = tf.compat.v1.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(tf.add(values, P_tmp), [-1]),
                                               default_value=0., validate_indices=False)
        N_reset = tf.compat.v1.sparse_to_dense(full_indices2, tf.shape(x), tf.reshape(tf.add(values2, N_tmp), [-1]),
                                               default_value=0., validate_indices=False)

        res = P_reset - N_reset

        return res

    def kSparse(self, x, topk):
        print('run regular k-sparse')
        dim = int(x.shape[1])
        print(x.shape)
        print(x)
        if topk > dim:
            warnings.warn('Warning: topk should not be larger than dim: %s, found: %s, using %s' % (dim, topk, dim))
            topk = dim

        k = dim - topk
        values, indices = tf.nn.top_k(-x, k)  # indices will be [[0, 1], [2, 1]], values will be [[6., 2.], [5., 4.]]

        # We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
        my_range = tf.expand_dims(tf.range(0, tf.shape(indices)[0]), 1)  # will be [[0], [1]]
        my_range_repeated = tf.tile(my_range, [1, k])  # will be [[0, 0], [1, 1]]

        full_indices = tf.stack([my_range_repeated, indices],
                                axis=2)  # change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
        full_indices = tf.reshape(full_indices, [-1, 2])

        to_reset = tf.compat.v1.sparse_to_dense(full_indices, tf.shape(x), tf.reshape(values, [-1]), default_value=0.,
                                                validate_indices=False)

        res = tf.add(x, to_reset)

        return res