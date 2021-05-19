import numpy as np


def model_batch_predictor(model, W, batch_size):
    """
    This function returns the predictions of the model on the input data W
    :param model:
    :param W:
    :param batch_size:
    :return: predictions
    """

    W = W.astype(np.float32)

    n_samples = W.shape[0]
    counter = 0
    pred = None
    while counter < n_samples // batch_size:
        _, curr_pred = \
            model.predict(W[batch_size * counter:batch_size * (counter + 1),
                          :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
        counter += 1
    if n_samples % batch_size != 0:
        _, curr_pred = \
            model.predict(W[batch_size * counter:, :].toarray())
        if counter:
            pred = np.vstack((pred, curr_pred))
        else:
            pred = curr_pred
    return pred


def batch_generator(W, A, beta_w, beta_a, batch_size, shuffle):
    """
    This function is the data generator used inside the fit() model

    :param W: the adjacency matrix
    :param A: the node attribute matrix
    :param beta_w: the hyper-parameter that regulates the reconstruction of W
    :param beta_a: the hyper-parameter that regulates the reconstruction of A
    :param batch_size: the batch size
    :param shuffle: whether to shuffle the data when an epoch ends
    :return: the input data for the fit() function
    """

    W = W.astype(np.float32)

    # compute the total number of batches
    number_of_batches = W.shape[0] // batch_size

    sample_index = np.arange(W.shape[0])

    # "counter" is an index used inside the while loop to control the data points to consider for each batch
    counter = 0
    if shuffle:
        np.random.shuffle(sample_index)
    while True:

        # batch_index is an array containing the indices of the elements considered for the batch
        batch_index = \
            sample_index[batch_size * counter:batch_size * (counter + 1)]

        # get the data points for the considered batch
        W_batch = W[batch_index, :].toarray()
        A_batch = A[batch_index, :]

        # this will be the input for the model in the fit() function
        InData = np.append(W_batch, A_batch, axis=1)

        # compute the values for the B parameter, which is used for the Hadmard product
        B_w = np.ones(W_batch.shape)
        B_w[W_batch != 0] = beta_w

        B_a = np.ones(A_batch.shape)
        B_a[A_batch != 0] = beta_a

        # OutData contains information used during the computation of the loss function
        OutData = [B_w, B_a, B_w]  # 3rd is a  dummy variable
        counter += 1

        # feed the fit() function with new data
        yield InData, OutData
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0
