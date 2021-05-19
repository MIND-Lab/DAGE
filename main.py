import tensorflow as tf
import os
import numpy as np
import time
import config

from model.model import DAGE
from dataset.load_datasets import load_dataset
from experiments.experiments import evaluate_node_classification_repeat


def main():

    if tf.test.is_built_with_cuda():
        print("[INFO] The GPU will be used for the computations.")
    else:
        print("[INFO] The GPU will NOT be used for the computations.")

    print("[INFO] Loading adjacency matrix, attribute matrix and class labels from files.")
    if config.dataset_name in ["citeseer-m10", "dblp"]:
        W, A, L = load_dataset("./dataset/" + config.dataset_name)
    else:
        raise Exception("Invalid dataset name: \"" + config.dataset_name + "\"")

    print("[INFO] Checking whether embeddings already exist...")
    if (os.path.exists(config.path_embedding)):
        print("[INFO] Loading embeddings from " + config.path_embedding + ".")
        embedding = np.loadtxt(config.path_embedding, delimiter=",")
        with open(config.path_embedding_id) as f:
            embedding_id = [line.rstrip('\n') for line in f]
        model = DAGE(d=config.d, Y=embedding, Y_id=embedding_id)
    else:
        print("[INFO] No pre-generated node embeddings could be found.")
        model = DAGE(d=config.d, lr=config.lr, beta_w=config.beta_w, beta_a=config.beta_a,
                      batch_size=config.batch_size, epochs=config.epochs, alpha=config.alpha,
                      beta=config.beta, gamma=config.gamma)

        print("[INFO] Learning the node embeddings (this might take a while, depending on the configurations)...")
        start = time.time()
        model.learn_embedding(W=W, A=A)
        end = time.time()
        print("[INFO] Successfully learned the node embeddings (took " + str(end - start) + " seconds).")
        embedding = model.get_embedding()
        embedding_id = model.get_embedding_id()
        model.save_embeddings(config.path_embedding, config.path_embedding_id)

    print("[INFO] Starting node classification evaluation")
    evaluate_node_classification_repeat(embedding, L)

if __name__ == '__main__':
    main()
