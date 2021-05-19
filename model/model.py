from keras.layers import Lambda, Subtract
from keras import backend as KBack
from keras.optimizers import Adam
from .autoencoders.autoencoders import *
from .utils import *
import os


class DAGE():
    """
    This class implements the DAGE model
    """

    def __init__(self, *hyper_dict, **kwargs):
        hyper_params = {
            'method_name': 'DAGE'
        }
        hyper_params.update(kwargs)
        for key in hyper_params.keys():
            self.__setattr__('_%s' % key, hyper_params[key])
        for dictionary in hyper_dict:
            for key in dictionary:
                self.__setattr__('_%s' % key, dictionary[key])

    def learn_embedding(self, W, A):
        """
        This function learns the embedding from the adjacency matrix and the attribute matrix

        :param W: adjacency matrix
        :param A: attribute matrix
        """

        self._node_num = W.shape[0]

        ## Input: a vector [a | b] where a is the i-th row of the adjacency matrix and b is the i-th attribute vector
        x_in = Input(shape=(self._node_num + A.shape[1],), name='x_in')
        w = Lambda(lambda x: x[:, 0:self._node_num], output_shape=(self._node_num,))(x_in)
        a = Lambda(lambda x: x[:, self._node_num:self._node_num + A.shape[1]], output_shape=(A.shape[1],))(x_in)

        ## build the autoencoder for preserving structural information
        structural_neurons_per_layer = [128]
        num_hidden_layers = len(structural_neurons_per_layer)
        structural_encoder = get_structural_encoder(self._node_num, self._d, num_hidden_layers,
                                                    structural_neurons_per_layer)
        structural_decoder = get_structural_decoder(self._node_num, self._d, num_hidden_layers,
                                                    structural_neurons_per_layer)
        structural_autoencoder = get_structural_autoencoder(structural_encoder, structural_decoder)

        ## build the autoencoder for preserving attribute information
        attribute_encoder, attr_encoder_layer = get_attribute_encoder(A.shape[1], self._d)
        attribute_decoder = get_attribute_decoder(self._d, A.shape[1], attr_encoder_layer)
        attribute_autoencoder = get_attribute_autoencoder(attribute_encoder, attribute_decoder)

        ## Process inputs
        [w_hat, y_w] = structural_autoencoder(w)
        [a_hat, y_a] = attribute_autoencoder(a)

        w_diff = Subtract()([w_hat, w])
        a_diff = Subtract()([a_hat, a])
        y_diff = Subtract()([y_w, y_a])

        model = Model(inputs=x_in, outputs=[w_diff, a_diff, y_diff])

        print("[INFO] Number of trainable parameters: %d" % model.count_params())

        # Objectives
        def second_order_proximity_loss(y_true, y_pred):
            '''
                y_pred: Contains W_hat - W
                y_true: Contains [B^W] for Hadamard product
            '''
            return KBack.sum(KBack.square(y_pred * y_true), axis=-1)

        def semantic_proximity_loss(y_true, y_pred):
            '''
                y_pred: Contains A_hat - A
                y_true: Contains [B^A] for Hadamard product
            '''
            return KBack.sum(KBack.square(y_pred * y_true), axis=-1)

        def consistency_loss(y_true, y_pred):
            '''
                y_pred: Contains y_a - y_w
            '''
            return KBack.sum(KBack.square(y_pred), axis=-1)

        model.compile(
            optimizer=Adam(lr=self._lr),
            loss=[second_order_proximity_loss, semantic_proximity_loss, consistency_loss],
            loss_weights=[self._alpha, self._beta, self._gamma]
        )

        model.fit(batch_generator(W, A, self._beta_w, self._beta_a, self._batch_size, True),
                  epochs=self._epochs,
                  steps_per_epoch=W.shape[0] // self._batch_size,
                  verbose=1,
        )

        self._Y = model_batch_predictor(structural_autoencoder, W, self._batch_size)
        self._Y_id = np.arange(A.shape[0])
        print("[INFO] Embedding dimension: %s" % str(self._Y.shape))
        self._decoder = structural_decoder

    def get_embedding_id(self):
        """
        This function returns the ids of the node embeddings

        :return: oredered node id vector associated to the node embeddings
        """
        return self._Y_id

    def get_embedding(self):
        """
        This function returns the node embeddings

        :return: node embeddings
        """
        return self._Y

    def save_embeddings(self, path_embedding, path_embedding_id):
        """
        This function is used to save the embeddings in a file

        :param path_embedding: the path where to save the embeddings
        :param path_embedding_id: the path where to save the embedding ids
        :return:
        """
        print("[INFO] Saving embedding to \"" + path_embedding[:path_embedding.rfind("/") + 1] + "\"...")
        if not os.path.isdir(path_embedding[:path_embedding.rfind("/") + 1]):
            os.mkdir(path_embedding[:path_embedding.rfind("/") + 1])
        np.savetxt(path_embedding, self._Y, delimiter=',')
        with open(path_embedding_id, 'w') as f:
            f.write("\n".join([str(i) for i in self._Y_id]))
        print("[INFO] Saving embedding to \"" + path_embedding[:path_embedding.rfind("/") + 1] + "\"... DONE")
