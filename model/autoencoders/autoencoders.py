from keras.models import Model
from keras.layers import Input, Dense
from .layers.dense_tied import Dense_tied
from .layers.kcompetitive import KCompetitive

def get_structural_encoder(node_num, d, num_hidden_layers, neurons_per_hidden_layer, activation_fn="sigmoid"):
    """
    This function generates the structural encoder.

    :param node_num: input dimension
    :param d: embedding dimension
    :param num_hidden_layers: number of hidden layers
    :param neurons_per_hidden_layer: neurons for each hidden layer
    :param activation_fn: activation function
    :return: the structural encoder
    """

    # Input
    x = Input(shape=(node_num,))
    # Encoder layers
    y = [None] * (num_hidden_layers + 1)
    y[0] = x  # y[0] is assigned the input
    for i in range(num_hidden_layers - 1):
        y[i + 1] = Dense(neurons_per_hidden_layer[i], activation=activation_fn)(y[i])
    y[num_hidden_layers] = Dense(d, activation=activation_fn)(y[num_hidden_layers - 1])
    # Encoder model
    encoder = Model(inputs=x, outputs=y[num_hidden_layers])
    return encoder


def get_structural_decoder(node_num, d, num_hidden_layers, neurons_per_hidden_layer, activation_fn="sigmoid"):
    """
    This function generates the structural decoder.

    :param node_num: input dimension
    :param d: embedding dimension
    :param num_hidden_layers: number of hidden layers
    :param neurons_per_hidden_layer: neurons for each hidden layer
    :param activation_fn: activation function
    :return: the structural decoder
    """

    # Input
    y = Input(shape=(d,))
    # Decoder layers
    y_hat = [None] * (num_hidden_layers + 1)
    y_hat[num_hidden_layers] = y
    for i in range(num_hidden_layers - 1, 0, -1):
        y_hat[i] = Dense(neurons_per_hidden_layer[i - 1], activation=activation_fn)(y_hat[i + 1])
    y_hat[0] = Dense(node_num, activation="sigmoid")(y_hat[1])
    # Output
    x_hat = y_hat[0]  # decoder's output is also the actual output
    # Decoder Model
    decoder = Model(inputs=y, outputs=x_hat)
    return decoder


def get_structural_autoencoder(encoder, decoder):
    """
    This function generates the structural autoencoder.

    :param encoder: a model for the encoder
    :param decoder: a model for the decoder
    :return: an autoencoder
    """

    # Input
    w = Input(shape=(encoder.layers[0].input_shape[0][1],))
    # Generate embedding
    h_w = encoder(w)
    # Generate reconstruction
    w_hat = decoder(h_w)
    # Autoencoder Model
    autoencoder = Model(inputs=w, outputs=[w_hat, h_w])
    return autoencoder


def get_attribute_encoder(num_nodes, d, activation_fn="sigmoid"):
    """
    This function generates the attribute encoder.

    :param num_nodes: input dimension
    :param d: embedding dimension
    :return: the attribute encoder
    """

    a = Input(shape=(num_nodes,))
    
    encoded_layer = Dense(d, activation=activation_fn, kernel_initializer="glorot_normal",
                          name="Encoded_Layer")
    encoded = encoded_layer(a)

    comp_topk = 128  # number of top k neurons with highest absolute value to consider in the KATE KCompetitive layer
    kfactor = 3  # multiplicative factor used inside the KATE KCompetitive layer
    kate_ctype = None # KATE embedding layer type

    encoded = KCompetitive(comp_topk, kate_ctype, kfactor)(encoded)

    encoder = Model(inputs=a, outputs=encoded)
    return encoder, encoded_layer



def get_attribute_decoder(d, num_nodes, encoded_layer):
    """
    This function generates the attribute decoder.

    :param d: embedding dimension
    :param num_nodes: input dimension
    :param encoded_layer: model used for encoder
    :return: the attribute decoder
    """

    y = Input(shape=(d,))

    # "decoded" is the lossy reconstruction of the input
    # add non-negativity contraint to ensure probabilistic interpretations
    decoded = Dense_tied(num_nodes, activation='sigmoid', tied_to=encoded_layer,
                         name='Decoded_Layer')(y)

    decoder = Model(inputs=y, outputs=decoded)
    return decoder



def get_attribute_autoencoder(encoder, decoder):
    """
    This function generates the attribute autoencoder.

    :param encoder: a model for the encoder
    :param decoder: a model for the decoder
    :return: an autoencoder
    """

    # Input
    a = Input(shape=(encoder.layers[0].input_shape[0][1],))
    # Generate embedding
    y_a = encoder(a)
    # Generate reconstruction
    a_hat = decoder(y_a)
    # Autoencoder Model
    autoencoder = Model(inputs=a, outputs=[a_hat, y_a])
    return autoencoder
